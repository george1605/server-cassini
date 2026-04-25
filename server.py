from flask import Flask, request, jsonify
from sentinelhub import (
    SHConfig, SentinelHubRequest, DataCollection,
    MimeType, BBox, CRS, MosaickingOrder
)
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import sys
import json
import logging
import threading
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

# ============================
# 🤖 CASSINI-AI INTEGRATION
# ============================
# Adaugă calea către cassini-ai pentru import
_AI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cassini-ai", "Apa")
sys.path.insert(0, _AI_DIR)

from calamity_ai.config import load_config as load_calamity_config
from calamity_ai.weather import get_open_meteo_weather_features, features_to_dict
from calamity_ai.scoring import score_calamities
from calamity_ai.context import get_environmental_context, environmental_context_to_dict
from calamity_ai.copernicus import get_copernicus_summary, copernicus_to_dict
from calamity_ai.forecast import get_open_meteo_predictions, predictions_to_dict
from calamity_ai.zones import get_zone_analysis, zone_analysis_to_dict
from calamity_ai.sensors import summarize_sensors
from calamity_ai.resources import ensure_resources, resource_summary_to_dict
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# =========================
# 🔐 CONFIG (folosește variabile de mediu!)
# =========================
# SH_ID = "sh-eb7762f1-3c97-47f4-95f0-d7be910e7d9a"
# SH_SECRET = "WKsvWEGUHnaJp6qKsFEtkAOGNs6GWVGz"

config = SHConfig()

config.sh_client_id = "sh-eb7762f1-3c97-47f4-95f0-d7be910e7d9a"
config.sh_client_secret = "WKsvWEGUHnaJp6qKsFEtkAOGNs6GWVGz"

print("ID:", config.sh_client_id)
print("SECRET:", config.sh_client_secret)

MODEL_PATH = "disaster_model.pkl"
SCALER_PATH = "scaler.pkl"

# ============================
# 📡 NODE.JS BACKEND CONNECTION
# ============================
# Schimbă URL-ul cu adresa backend-ului Node.js al colegului
NODE_BACKEND_URL = os.environ.get("NODE_BACKEND_URL", "http://localhost:3001")

def forward_to_node_backend(report, report_type="weather-risk"):
    """
    Trimite raportul AI la backend-ul Node.js.
    Rulează într-un thread separat ca să nu blocheze Flask.
    """
    def _send():
        try:
            payload = {
                "source": "cassini-ai",
                "type": report_type,
                "received_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "data": report
            }
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{NODE_BACKEND_URL}/api/alerts",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                logger.info(f"✅ Alert forwarded to Node.js → {resp.status}")
        except urllib.error.URLError as e:
            logger.warning(f"⚠️  Node.js backend unreachable ({NODE_BACKEND_URL}): {e}")
        except Exception as e:
            logger.error(f"❌ Forward to Node.js failed: {e}")

    threading.Thread(target=_send, daemon=True).start()
# =========================
# 🛰️ EVALSCRIPT MULTI-INDEX
# =========================
# Calculează mai mulți indici spectrali pentru detectare dezastre
evalscript_multi_index = """
//VERSION=3
function setup() {
    return {
        input: ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"],
        output: [
            { id: "indices", bands: 6, sampleType: "FLOAT32" },
            { id: "rgb", bands: 3, sampleType: "UINT8" }
        ],
        mosaicking: "ORBIT"
    };
}
function validate(sample) {
    // Exclude clouds, shadows, no data
    const invalid = [0, 1, 3, 8, 9, 10];
    return !invalid.includes(sample.SCL);
}
function evaluatePixel(samples) {
    // Filtrează pixelii cu nori
    var valid = samples.filter(validate);
    if (valid.length === 0) {
        return {
            indices: [-999, -999, -999, -999, -999, -999],
            rgb: [0, 0, 0]
        };
    }
    
    var s = valid[0];
    
    // NDVI - vegetație (scade după incendii/defrișări)
    var ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 0.0001);
    
    // NDWI - apă (crește în inundații)
    var ndwi = (s.B03 - s.B08) / (s.B03 + s.B08 + 0.0001);
    
    // NDMI - umiditate vegetație
    var ndmi = (s.B08 - s.B11) / (s.B08 + s.B11 + 0.0001);
    
    // NBR - zone arse (scade după incendii)
    var nbr = (s.B08 - s.B12) / (s.B08 + s.B12 + 0.0001);
    
    // BSI - sol gol (crește după alunecări de teren)
    var bsi = ((s.B11 + s.B04) - (s.B08 + s.B02)) / 
              ((s.B11 + s.B04) + (s.B08 + s.B02) + 0.0001);
    
    // NDBI - zone construite
    var ndbi = (s.B11 - s.B08) / (s.B11 + s.B08 + 0.0001);
    
    return {
        indices: [ndvi, ndwi, ndmi, nbr, bsi, ndbi],
        rgb: [2.5 * s.B04 * 255, 2.5 * s.B03 * 255, 2.5 * s.B02 * 255]
    };
}
"""
# =========================
# 📥 FETCH SATELLITE DATA
# =========================
def get_satellite_data(bbox_coords, start_date, end_date, resolution=64):
    print("Requesting data...")

    bbox = BBox(bbox_coords, crs=CRS.WGS84)

    request = SentinelHubRequest(
        evalscript=evalscript_multi_index,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(start_date, end_date),
                mosaicking_order=MosaickingOrder.LEAST_CC
            )
        ],
        responses=[
            SentinelHubRequest.output_response("indices", MimeType.TIFF),
            SentinelHubRequest.output_response("rgb", MimeType.PNG)
        ],
        bbox=bbox,
        size=(resolution, resolution),
        config=config   # 🔥 FOLOSEȘTE CONFIGUL GLOBAL
    )

    return request.get_data()
# =========================
# 🧠 FEATURE EXTRACTION
# =========================
def extract_features(indices_img):
    """
    Extrage features din imaginea cu indici spectrali.
    indices_img: array (H, W, 6) cu [NDVI, NDWI, NDMI, NBR, BSI, NDBI]
    """
    features = []
    index_names = ['ndvi', 'ndwi', 'ndmi', 'nbr', 'bsi', 'ndbi']
    
    for i, name in enumerate(index_names):
        band = indices_img[:, :, i]
        # Exclude no-data values (-999)
        valid = band[band > -900]
        
        if len(valid) == 0:
            features.extend([0] * 7)
            continue
        
        features.extend([
            float(np.mean(valid)),
            float(np.std(valid)),
            float(np.min(valid)),
            float(np.max(valid)),
            float(np.percentile(valid, 25)),
            float(np.percentile(valid, 75)),
            float(np.sum(valid > 0) / len(valid))  # procent valori pozitive
        ])
    
    # Features adiționale: rapoarte între indici
    means = features[::7][:6]  # extrage mean-urile
    if means[0] != 0:  # NDVI mean
        features.append(means[3] / (abs(means[0]) + 0.001))  # NBR/NDVI ratio
    else:
        features.append(0)
    
    return features
def extract_change_features(before_img, after_img):
    """
    Detectează schimbări între două perioade.
    Util pentru detectare dezastre prin comparație temporală.
    """
    features_before = extract_features(before_img)
    features_after = extract_features(after_img)
    
    # Diferențe absolute și relative
    diff_features = []
    for b, a in zip(features_before, features_after):
        diff_features.append(a - b)  # diferență absolută
        if abs(b) > 0.001:
            diff_features.append((a - b) / abs(b))  # diferență relativă
        else:
            diff_features.append(0)
    
    return features_after + diff_features
# =========================
# 📊 DISASTER TYPES
# =========================
DISASTER_TYPES = {
    0: "normal",
    1: "flood",        # inundație
    2: "fire",         # incendiu
    3: "deforestation",# defrișare
    4: "landslide"     # alunecare de teren
}
# =========================
# 🏋️ TRAIN MODEL
# =========================
@app.route("/train", methods=["POST"])
def train():
    """
    Antrenează modelul cu date reale.
    
    Trimite JSON cu structura:
    {
        "samples": [
            {
                "bbox": [minx, miny, maxx, maxy],
                "date_before": "2024-01-01",
                "date_after": "2024-02-01",
                "label": 1  // 0=normal, 1=flood, 2=fire, etc.
            },
            ...
        ]
    }
    """
    data = request.json
    samples = data.get("samples", [])
    
    if len(samples) < 10:
        return jsonify({
            "error": "Ai nevoie de cel puțin 10 sample-uri pentru antrenare"
        }), 400
    
    X = []
    y = []
    
    for i, sample in enumerate(samples):
        try:
            logger.info(f"Procesez sample {i+1}/{len(samples)}")
            
            bbox = sample["bbox"]
            
            # Descarcă date înainte de eveniment
            before_data = get_satellite_data(
                bbox,
                sample["date_before"],
                sample.get("date_before_end", sample["date_before"])
            )
            
            # Descarcă date după eveniment
            after_data = get_satellite_data(
                bbox,
                sample["date_after"],
                sample.get("date_after_end", sample["date_after"])
            )
            
            # Extrage features cu change detection
            features = extract_change_features(
                before_data[0],  # indices before
                after_data[0]    # indices after
            )
            
            X.append(features)
            y.append(sample["label"])
            
        except Exception as e:
            logger.error(f"Eroare la sample {i}: {e}")
            continue
    
    if len(X) < 5:
        return jsonify({"error": "Prea puține sample-uri procesate cu succes"}), 400
    
    X = np.array(X)
    y = np.array(y)
    
    # Split pentru validare
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Antrenare Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluare
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Salvare model
    joblib.dump(model, MODEL_PATH)
    
    # Feature importance
    importance = dict(zip(
        [f"feature_{i}" for i in range(len(model.feature_importances_))],
        model.feature_importances_.tolist()
    ))
    
    return jsonify({
        "status": "Model antrenat cu succes",
        "samples_used": len(X),
        "accuracy": report.get("accuracy", 0),
        "classification_report": report,
        "top_features": dict(sorted(
            importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10])
    })
# =========================
# 🔮 PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    """
    Predicție pentru o zonă nouă.
    
    JSON input:
    {
        "bbox": [minx, miny, maxx, maxy],
        "date_before": "2024-01-01",
        "date_after": "2024-02-01"
    }
    """
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Modelul nu e antrenat. Apelează /train mai întâi"}), 400
    
    data = request.json
    bbox = data.get("bbox")
    date_before = data.get("date_before")
    date_after = data.get("date_after")
    
    if not all([bbox, date_before, date_after]):
        return jsonify({"error": "Lipsesc parametri: bbox, date_before, date_after"}), 400
    
    try:
        # Descarcă date
        before_data = get_satellite_data(bbox, date_before, date_before)
        after_data = get_satellite_data(bbox, date_after, date_after)
        
        # Extrage features
        features = extract_change_features(before_data[0], after_data[0])
        
        # Predicție
        model = joblib.load(MODEL_PATH)
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        
        return jsonify({
            "prediction": int(prediction),
            "disaster_type": DISASTER_TYPES.get(int(prediction), "unknown"),
            "confidence": float(max(probabilities)),
            "probabilities": {
                DISASTER_TYPES.get(i, f"class_{i}"): float(p) 
                for i, p in enumerate(probabilities)
            },
            "bbox": bbox,
            "period": {
                "before": date_before,
                "after": date_after
            }
        })
        
    except Exception as e:
        logger.error(f"Eroare la predicție: {e}")
        return jsonify({"error": str(e)}), 500
# =========================
# 📊 ANALYZE (fără ML)
# =========================
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analiză bazată pe threshold-uri, fără ML.
    Util pentru detectare rapidă sau când nu ai model antrenat.
    """
    data = request.json
    bbox = data.get("bbox")
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    
    try:
        satellite_data = get_satellite_data(bbox, start_date, end_date)
        indices = satellite_data[0]
        
        # Extrage indici individuali
        valid_mask = indices[:, :, 0] > -900
        
        ndvi = indices[:, :, 0][valid_mask]
        ndwi = indices[:, :, 1][valid_mask]
        nbr = indices[:, :, 3][valid_mask]
        bsi = indices[:, :, 4][valid_mask]
        
        alerts = []
        
        # Detectare inundații (NDWI ridicat)
        flood_pixels = np.sum(ndwi > 0.3) / len(ndwi) if len(ndwi) > 0 else 0
        if flood_pixels > 0.1:
            alerts.append({
                "type": "flood",
                "severity": "high" if flood_pixels > 0.3 else "medium",
                "affected_area_percent": round(flood_pixels * 100, 2)
            })
        
        # Detectare incendii (NDVI scăzut + NBR scăzut)
        burned_pixels = np.sum((ndvi < 0.1) & (nbr < -0.1)) / len(ndvi) if len(ndvi) > 0 else 0
        if burned_pixels > 0.05:
            alerts.append({
                "type": "fire_damage",
                "severity": "high" if burned_pixels > 0.2 else "medium",
                "affected_area_percent": round(burned_pixels * 100, 2)
            })
        
        # Detectare sol gol/alunecare (BSI ridicat)
        bare_soil = np.sum(bsi > 0.2) / len(bsi) if len(bsi) > 0 else 0
        if bare_soil > 0.15:
            alerts.append({
                "type": "landslide_or_deforestation",
                "severity": "high" if bare_soil > 0.4 else "medium",
                "affected_area_percent": round(bare_soil * 100, 2)
            })
        
        result = {
            "bbox": bbox,
            "period": {"start": start_date, "end": end_date},
            "statistics": {
                "ndvi_mean": float(np.mean(ndvi)) if len(ndvi) > 0 else None,
                "ndwi_mean": float(np.mean(ndwi)) if len(ndwi) > 0 else None,
                "nbr_mean": float(np.mean(nbr)) if len(nbr) > 0 else None,
                "valid_pixels_percent": round(np.sum(valid_mask) / valid_mask.size * 100, 2)
            },
            "alerts": alerts,
            "alert_count": len(alerts)
        }

        # 📡 Forward la Node.js
        forward_to_node_backend(result, report_type="satellite-analyze")

        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Eroare la analiză: {e}")
        return jsonify({"error": str(e)}), 500

# ============================
# 🤖 CASSINI-AI: CONFIG
# ============================
_CALAMITY_CONFIG_PATH = os.path.join(_AI_DIR, "config", "monitor_config.json")
_SENSORS_PATH = os.path.join(_AI_DIR, "data", "sensors.csv")

try:
    calamity_config = load_calamity_config(_CALAMITY_CONFIG_PATH)
    logger.info("✅ Cassini-AI config loaded: %s", calamity_config.area_name)
except Exception as e:
    logger.warning("⚠️  Cassini-AI config not loaded: %s", e)
    calamity_config = None

# ============================
# 🌦️ WEATHER RISK – FULL
# ============================
@app.route("/weather-risk", methods=["POST"])
def weather_risk():
    """
    Raport complet de risc meteo din cassini-ai.

    JSON input (totul opțional):
    {
        "skip_copernicus": false,
        "skip_context": false,
        "skip_zones": false,
        "skip_predictions": false,
        "prediction_days": 5
    }
    """
    if calamity_config is None:
        return jsonify({"error": "Cassini-AI config not loaded"}), 500

    data = request.json or {}
    skip_copernicus = data.get("skip_copernicus", False)
    skip_context = data.get("skip_context", False)
    skip_zones = data.get("skip_zones", False)
    skip_predictions = data.get("skip_predictions", False)
    prediction_days = data.get("prediction_days", 5)

    try:
        now = datetime.now(timezone.utc)

        # 1. Weather features de la Open-Meteo
        features = get_open_meteo_weather_features(calamity_config)

        # 2. Context istoric + elevație
        context = None
        if not skip_context:
            try:
                context = get_environmental_context(calamity_config, now=now)
            except Exception as ctx_err:
                logger.warning("Context fetch failed: %s", ctx_err)

        # 3. Scoring calamități
        calamities = score_calamities(features, calamity_config.thresholds, context=context)

        # 4. Zone analysis
        zone_analysis = None
        if context and not skip_zones:
            try:
                zone_analysis = get_zone_analysis(
                    calamity_config,
                    features=features,
                    calamities=calamities,
                    context=context,
                )
            except Exception as z_err:
                logger.warning("Zone analysis failed: %s", z_err)

        # 5. Predicții multi-day
        predictions = None
        if context and not skip_predictions:
            try:
                predictions = get_open_meteo_predictions(
                    calamity_config, context=context, days=prediction_days
                )
            except Exception as p_err:
                logger.warning("Predictions failed: %s", p_err)

        # 6. Copernicus satellite
        copernicus = None
        if not skip_copernicus:
            try:
                copernicus = get_copernicus_summary(calamity_config, now=now)
            except Exception as c_err:
                logger.warning("Copernicus failed: %s", c_err)

        # 7. Sensors health
        sensors = summarize_sensors(
            _SENSORS_PATH,
            now=now,
            min_online_ratio=calamity_config.sensor_health["min_online_ratio"],
            stale_after_minutes=calamity_config.sensor_health["stale_after_minutes"],
        )

        # Build report
        report = {
            "timestamp": now.isoformat().replace("+00:00", "Z"),
            "area": calamity_config.area_name,
            "weather": features_to_dict(features),
            "calamities": calamities,
            "sensors": {
                "total": sensors.total,
                "online": sensors.online,
                "offline": sensors.offline,
                "stale": sensors.stale,
                "working": sensors.working,
            },
        }
        if copernicus:
            report["copernicus"] = copernicus_to_dict(copernicus)
        if context:
            report["context"] = environmental_context_to_dict(context)
        if zone_analysis:
            report["zone_analysis"] = zone_analysis_to_dict(zone_analysis)
        if predictions:
            report["predictions"] = predictions_to_dict(predictions)

        # 📡 Forward la Node.js
        forward_to_node_backend(report, report_type="weather-risk-full")

        return jsonify(report)

    except Exception as e:
        logger.error(f"Eroare la weather-risk: {e}")
        return jsonify({"error": str(e)}), 500

# ============================
# ⚡ WEATHER RISK – QUICK
# ============================
@app.route("/weather-risk/quick", methods=["GET"])
def weather_risk_quick():
    """
    Raport rapid de risc meteo – doar weather + scoring, fără Copernicus/context.
    Nu necesită parametri.
    """
    if calamity_config is None:
        return jsonify({"error": "Cassini-AI config not loaded"}), 500

    try:
        now = datetime.now(timezone.utc)
        features = get_open_meteo_weather_features(calamity_config)
        calamities = score_calamities(features, calamity_config.thresholds)

        quick_report = {
            "timestamp": now.isoformat().replace("+00:00", "Z"),
            "area": calamity_config.area_name,
            "weather": features_to_dict(features),
            "calamities": calamities,
        }

        # 📡 Forward la Node.js
        forward_to_node_backend(quick_report, report_type="weather-risk-quick")

        return jsonify(quick_report)

    except Exception as e:
        logger.error(f"Eroare la weather-risk/quick: {e}")
        return jsonify({"error": str(e)}), 500

# =========================
# 🩺 HEALTH CHECK
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_trained": os.path.exists(MODEL_PATH),
        "config_valid": bool(config.sh_client_id and config.sh_client_secret),
        "cassini_ai_loaded": calamity_config is not None,
        "cassini_ai_area": calamity_config.area_name if calamity_config else None
    })
# =========================
# 🚀 RUN
# =========================
if __name__ == "__main__":
    if not config.sh_client_id or not config.sh_client_secret:
        logger.warning("⚠️  Setează variabilele SH_CLIENT_ID și SH_CLIENT_SECRET!")
    
    app.run(debug=True, host="0.0.0.0", port=5000)