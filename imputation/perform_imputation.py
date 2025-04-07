# predict_imputations.py

import os
import torch
import rasterio
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
from imputation_model import SentinelImputer
from imputation_model import save_geotiff

# --- New Dataset for Inference ---
class MODISOnlyDataset(Dataset):
    def __init__(self, modis_dirs):
        self.modis_dirs = modis_dirs

    def __len__(self):
        return len(self.modis_dirs)

    def __getitem__(self, idx):
        modis_stack = []
        modis_path = self.modis_dirs[idx]
        for i in range(1, 8):
            with rasterio.open(os.path.join(modis_path, f'band{i}.tif')) as src:
                band = src.read(1).astype(np.float32)
                modis_stack.append(band)
                if i == 1:
                    transform = src.transform
                    crs = src.crs
        modis = np.stack(modis_stack)
        return {
            "modis": torch.tensor(modis),
            "transform": transform,
            "crs": crs,
            "filename": os.path.basename(modis_path)
        }

# --- Inference Function ---
def run_inference(modis_root, model_path, output_dir="outputs"):
    modis_dirs = sorted([d for d in glob(os.path.join(modis_root, "*")) if os.path.isdir(d)])
    dataset = MODISOnlyDataset(modis_dirs)
    loader = DataLoader(dataset, batch_size=1)

    model = SentinelImputer.load_from_checkpoint(model_path, in_ch=7, out_ch=12)
    model.eval()
    model.freeze()

    for batch in loader:
        modis = batch["modis"]
        with torch.no_grad():
            pred = model(modis)

        save_geotiff(
            pred[0],  # remove batch dimension
            batch["transform"][0],
            batch["crs"][0],
            batch["filename"][0] + "_imputed.tif",
            output_dir=output_dir
        )
        print(f"Saved imputed Sentinel-2 image for {batch['filename'][0]}")

# --- Run if script is called directly ---
if __name__ == "__main__":
    MODIS_FOLDER = "./modis"
    MODEL_CHECKPOINT = "./models/best-checkpoint.ckpt"
    OUTPUT_FOLDER = "./imputed_s2"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Run inference
    run_inference(MODIS_FOLDER, MODEL_CHECKPOINT, OUTPUT_FOLDER)
