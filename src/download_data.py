import os, gdown

MODEL_ID = "13U8ENJ4Rci1dkqljbkThvdGwEf8xyS9v"
TEST_DATA_ID = "1SPIKUoneEZih6AuXQFg5_PyD_Lc9P0IE"

files = {
    'models/convlstm_best.keras': MODEL_ID,
    'data/test_data.npz': TEST_DATA_ID,
}

def download_all():
    for path, fid in files.items():
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            url = f"https://drive.google.com/uc?id={fid}"
            gdown.download(url, path, quiet=False)