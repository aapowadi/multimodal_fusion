"""
Goal : Train a U-Net model for imputation of Sent-1 and Sent-2 images using MODIS images.
Input : MODIS images, Sent-1 and Sent-2 images
Output : Trained U-Net model for imputation saved as a .pth file ./models
"""

"""
Use PyTorch Lightning for training the model. PyTorch Lightning is a lightweight wrapper around PyTorch that helps with organizing PyTorch code and provides features like automatic checkpointing, logging, and distributed training.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pytorch_lightning as pl
import rasterio
from rasterio.transform import from_origin
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from pytorch_lightning.callbacks import ModelCheckpoint
from glob import glob

# --- Dataset Class ---
class SentinelImputeDataset(Dataset):
    def __init__(self, modis_dirs, s2_dirs):
        self.modis_dirs = modis_dirs
        self.s2_dirs = s2_dirs

    def __len__(self):
        return len(self.modis_dirs)

    def __getitem__(self, idx):
        modis_stack = []
        for i in range(1, 8):
            with rasterio.open(os.path.join(self.modis_dirs[idx], f'band{i}.tif')) as src:
                modis_stack.append(src.read(1).astype(np.float32))
                if i == 1:
                    transform = src.transform
                    crs = src.crs

        s2_stack = []
        for i in range(1, 13):
            with rasterio.open(os.path.join(self.s2_dirs[idx], f'band{i}.tif')) as src:
                s2_stack.append(src.read(1).astype(np.float32))

        modis = np.stack(modis_stack)
        s2 = np.stack(s2_stack)

        mask = (~np.isnan(s2) & (s2 != 0)).astype(np.float32)
        s2_clean = np.nan_to_num(s2, nan=0.0)

        return {
            'modis': torch.tensor(modis),
            's2': torch.tensor(s2_clean),
            'mask': torch.tensor(mask),
            'transform': transform,
            'crs': crs,
            'filename': os.path.basename(self.s2_dirs[idx])
        }

# --- DataLoader Setup ---
def get_dataloaders(modis_root, s2_root, batch_size=4):
    modis_dirs = sorted([d for d in glob(os.path.join(modis_root, '*')) if os.path.isdir(d)])
    s2_dirs = sorted([d for d in glob(os.path.join(s2_root, '*')) if os.path.isdir(d)])

    train_m, val_m, train_s2, val_s2 = train_test_split(modis_dirs, s2_dirs, test_size=0.2, random_state=42)

    train_ds = SentinelImputeDataset(train_m, train_s2)
    val_ds = SentinelImputeDataset(val_m, val_s2)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl

# --- UNet Model ---
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.Conv2d(64, out_channels, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# --- Lightning Module ---
class SentinelImputer(pl.LightningModule):
    def __init__(self, in_ch=7, out_ch=12, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(in_ch, out_ch)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pred = self(batch['modis'])
        loss = self.loss_fn(pred * batch['mask'], batch['s2'] * batch['mask'])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch['modis'])
        loss = self.loss_fn(pred * batch['mask'], batch['s2'] * batch['mask'])
        self.log("val_loss", loss, prog_bar=True)

        pred_np = pred.detach().cpu().numpy()
        gt_np = batch['s2'].cpu().numpy()
        mask_np = batch['mask'].cpu().numpy()
        ssim_total = 0
        for b in range(pred_np.shape[0]):
            for ch in range(pred_np.shape[1]):
                gt_band = gt_np[b, ch] * mask_np[b, ch]
                pred_band = pred_np[b, ch] * mask_np[b, ch]
                data_range = (gt_band.max() - gt_band.min()) or 1.0
                ssim_val = ssim(gt_band, pred_band, data_range=data_range)
                ssim_total += ssim_val
        ssim_avg = ssim_total / (pred_np.shape[0] * pred_np.shape[1])
        self.log("val_ssim", ssim_avg, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

# --- Save Prediction as GeoTIFF ---
def save_geotiff(pred, transform, crs, filename, output_dir="./imputed_images"):
    os.makedirs(output_dir, exist_ok=True)
    pred_np = pred.detach().cpu().numpy()
    path = os.path.join(output_dir, filename)
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=pred_np.shape[1], width=pred_np.shape[2], count=pred_np.shape[0],
        dtype=pred_np.dtype, crs=crs, transform=transform
    ) as dst:
        dst.write(pred_np)

# --- Visualization Function ---
def plot_sample(batch, pred, idx=0, band=3):
    gt = batch['s2'][idx, band].numpy()
    pr = pred[idx, band].detach().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(gt, cmap='viridis')
    axs[0].set_title("Ground Truth")
    axs[1].imshow(pr, cmap='viridis')
    axs[1].set_title("Predicted")
    plt.tight_layout()
    plt.show()

# --- Training Script ---
if __name__ == "__main__":
    modis_dir = "data/modis"
    s2_dir = "data/s2"

    train_loader, val_loader = get_dataloaders(modis_dir, s2_dir)

    model = SentinelImputer(in_ch=7, out_ch=12)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="models/",
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
        save_weights_only=True,
    )

    trainer = pl.Trainer(max_epochs=20, accelerator="auto", callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
    
    
    # Plot and save prediction
    batch = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        pred = model(batch['modis'])
    plot_sample(batch, pred, idx=0, band=3)

    '''
    for i in range(pred.shape[0]):
        save_geotiff(pred[i], batch['transform'][i], batch['crs'][i], batch['filename'][i])
    '''
