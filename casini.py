from sentinelhub import (
    SHConfig, SentinelHubRequest, DataCollection,
    MimeType, BBox, CRS, Geometry
)

# 1. Configurarea autentificării (Pune datele tale din Dashboard)
config = SHConfig()
config.sh_client_id = 'INTRODU_CLIENT_ID_AICI'
config.sh_client_secret = 'INTRODU_CLIENT_SECRET_AICI'

# 2. Definirea zonei (Exemplu: Zona de sud, inundabilă, a Dunării)
# Format: [Lon_Min, Lat_Min, Lon_Max, Lat_Max]
bbox_romania_sud = BBox(bbox=[22.50, 43.70, 24.50, 44.50], crs=CRS.WGS84)

# 3. Evalscript pentru Sentinel-1 (VV pentru detectarea apei)
# Acest script returnează valorile brute în format FLOAT32, ideale pentru AI
evalscript_s1 = """
//VERSION=3
function setup() {
  return {
    input: ["VV"],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluate(sample) {
  return [sample.VV];
}
"""

# 4. Construirea cererii
request = SentinelHubRequest(
    evalscript=evalscript_s1,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL1_IW,
            time_interval=('2023-10-01', '2023-10-10') # Ajustează data aici
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.TIFF)
    ],
    bbox=bbox_romania_sud,
    size=[1024, 768], # Rezoluția imaginii de ieșire
    config=config
)

# 5. Descărcarea datelor
data = request.get_data()

# Rezultatul 'data[0]' este un array NumPy gata pentru modelul tău de AI
print(f"Imagine descărcată cu succes. Shape: {data[0].shape}")