import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score
import numpy as np
import os
from datetime import datetime
import rasterio
import rasterio.windows
import pandas as pd
import matplotlib.pyplot as plt
import logging
import pdb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiModalDataset(Dataset):
    def __init__(self, sentinel1_dir, sentinel2_dir, modis_dir, crop_dir, soil_dir, weather_dir, transform=None):
        self.sentinel1_dir = sentinel1_dir
        self.sentinel2_dir = sentinel2_dir
        self.modis_dir = modis_dir
        self.cdl_dir = crop_dir
        self.soil_dir = soil_dir
        self.weather_dir = weather_dir
        self.transform = transform
        self.patch_size = 16
        self.delta_lat = 0.01446

        self.week_start_dates = [
            d for d in os.listdir(sentinel1_dir)
            if os.path.isdir(os.path.join(sentinel1_dir, d)) and self._is_in_april_to_september(d) 
        ]
        if not self.week_start_dates:
            raise ValueError("No Sentinel-1 data found for April to September.")

        self.s1_bands = ['vv', 'vh']
        self.s2_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'] # 12 bands + 'B10' which is missing sometimes.
        self.modis_bands = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7']
        self.weather_bands = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
        self.cdl_bands = ['Band_1']
        self.soil_bands = ['aws100', 'aws150', 'aws999', 'nccpi3all', 'nccpi3corn', 'rootznaws', 'soc150', 'soc999', 'pctearthmc']

        sample_s1_path = os.path.join(sentinel1_dir, self.week_start_dates[0], '4326_vv.tif')
        with rasterio.open(sample_s1_path) as src:
            self.bounds = src.bounds
            self.transform_geo = src.transform
            self.width, self.height = src.width, src.height

    def _is_in_april_to_september(self, date_str):
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            month = date.month
            day = date.day
            return (month == 4 and day >= 1) or (4 < month < 9) or (month == 9 and day <= 23)
        except ValueError:
            return False

    def _load_and_crop_single_band(self, path, bbox, cdl=False, band_name=None, stats=None):
        try:
            with rasterio.open(path) as src:
                window = rasterio.windows.from_bounds(*bbox, transform=src.transform)
                data = src.read(1, window=window, boundless=True, fill_value=np.nan)
                data_tensor = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
                interpolation = F.InterpolationMode.NEAREST_EXACT if cdl else F.InterpolationMode.BILINEAR
                resized = F.resize(data_tensor, [self.patch_size, self.patch_size], interpolation=interpolation, antialias=True)
                return resized.squeeze(0)
        except Exception as e:
            logger.warning(f"Failed to load band {band_name} from {path}: {e}. Generating dummy data.")
            # Generate dummy data
            if stats and band_name in stats:
                vmin, vmax = stats[band_name]
                dummy_value = (vmin + vmax) / 2
            else:
                dummy_value = 0.0
            return torch.full((self.patch_size, self.patch_size), dummy_value, dtype=torch.float32)

    def _random_patch_coords(self):
        sample_path = os.path.join(self.sentinel1_dir, self.week_start_dates[0], '4326_vv.tif')
        with rasterio.open(sample_path) as src:
            bounds = src.bounds
        
        lon_min = bounds.left + self.delta_lat / 2
        lon_max = bounds.right - self.delta_lat / 2
        lat_min = bounds.bottom + self.delta_lat / 2
        lat_max = bounds.top - self.delta_lat / 2
        
        lat_center = np.random.uniform(lat_min, lat_max)
        lon_center = np.random.uniform(lon_min, lon_max)
        
        delta_lat = self.delta_lat
        delta_lon = delta_lat / np.cos(np.radians(lat_center))
        
        left = lon_center - delta_lon / 2
        right = lon_center + delta_lon / 2
        bottom = lat_center - delta_lat / 2
        top = lat_center + delta_lat / 2
        
        return left, bottom, right, top

    def __len__(self):
        return len(self.week_start_dates)

    def __getitem__(self, idx):
        week_start_date = self.week_start_dates[idx]
        year = week_start_date.split('-')[0]
        bbox = self._random_patch_coords()

        s1_min_max, s2_min_max, modis_min_max, soil_min_max, weather_min_max = load_statistics()

        # Sentinel-1
        s1_folder = os.path.join(self.sentinel1_dir, week_start_date)
        s1_patches = [
            self._load_and_crop_single_band(
                os.path.join(s1_folder, f'4326_{band}.tif'), bbox, band_name=band, stats=s1_min_max
            ) for band in self.s1_bands
        ]
        s1_tensor = torch.stack(s1_patches, dim=0)

        # Sentinel-2
        s2_folder = os.path.join(self.sentinel2_dir, week_start_date)
        s2_patches = [
            self._load_and_crop_single_band(
                os.path.join(s2_folder, f'4326_{band}.tif'), bbox, band_name=band, stats=s2_min_max
            ) for band in self.s2_bands
        ]
        s2_tensor = torch.stack(s2_patches, dim=0)

        # MODIS
        modis_folder = os.path.join(self.modis_dir, week_start_date)
        modis_patches = [
            self._load_and_crop_single_band(
                os.path.join(modis_folder, f'4326_{band}.tif'), bbox, band_name=band, stats=modis_min_max
            ) for band in self.modis_bands
        ]
        modis_tensor = torch.stack(modis_patches, dim=0)

        # CDL (used for label)
        crop_path = os.path.join(self.cdl_dir, f'{year}_WGS84.tif')
        try:
            cdl_tensor = self._load_and_crop_single_band(crop_path, bbox, cdl=True, band_name='Band_1')
            cdl_flat = cdl_tensor.flatten()
            label = torch.mode(cdl_flat).values.item()
        except Exception as e:
            logger.error(f"Failed to load CDL data {crop_path}: {e}. Assigning dummy label.")
            label = 0  # Dummy label; adjust based on dataset

        # Soil
        soil_patches = [
            self._load_and_crop_single_band(
                os.path.join(self.soil_dir, f'merged_max_{band}_resampled_new.tif'), bbox, band_name=band, stats=soil_min_max
            ) for band in self.soil_bands
        ]
        soil_tensor = torch.stack(soil_patches, dim=0)

        # Weather
        weather_patches = []
        date = datetime.strptime(week_start_date, '%Y-%m-%d')
        doy = date.timetuple().tm_yday
        for band in self.weather_bands:
            week_stack = []
            for i in range(doy, doy + 7):
                date_s = datetime.strptime(f'{year}-{i}', '%Y-%j')
                date_str = date_s.strftime('%Y-%m-%d')
                path = os.path.join(self.weather_dir, band, f'{date_str}.tif')
                try:
                    patch = self._load_and_crop_single_band(path, bbox, band_name=band, stats=weather_min_max)
                    week_stack.append(patch)
                except Exception as e:
                    logger.warning(f"Skipping missing weather band {band} for {date_str}: {e}")
                    continue
            if week_stack:
                week_avg = torch.nanmean(torch.stack(week_stack, dim=0), dim=0)
            else:
                logger.warning(f"All daily files missing for weather band {band}. Generating dummy data.")
                dummy_value = weather_min_max.get(band, (0.0, 0.0))[0] if weather_min_max else 0.0
                week_avg = torch.full((self.patch_size, self.patch_size), dummy_value, dtype=torch.float32)
            weather_patches.append(week_avg)
        weather_tensor = torch.stack(weather_patches, dim=0)

        modalities = [s1_tensor, s2_tensor, modis_tensor, soil_tensor, weather_tensor]

        if self.transform:
            modalities = [self.transform(m) for m in modalities]

        return modalities, label

def load_statistics(s1_path='s1_statistics.csv', s2_path='s2_statistics.csv', modis_path='modis_statistics.csv',
                    soil_path='soil_statistics.csv', weather_path='weather_statistics.csv'):
    try:
        s1_stats = pd.read_csv(os.path.join('statistics', s1_path))
        s2_stats = pd.read_csv(os.path.join('statistics', s2_path))
        modis_stats = pd.read_csv(os.path.join('statistics', modis_path))
        soil_stats = pd.read_csv(os.path.join('statistics', soil_path))
        weather_stats = pd.read_csv(os.path.join('statistics', weather_path))
    except FileNotFoundError as e:
        logger.error(f"Could not find one of the CSV files: {e}")
        return None, None, None, None, None
    
    s1_min_max = {row['Band']: (row['Min'], row['Max']) for _, row in s1_stats.iterrows()}
    s2_min_max = {row['Band']: (row['Min'], row['Max']) for _, row in s2_stats.iterrows()}
    modis_min_max = {row['Band']: (row['Min'], row['Max']) for _, row in modis_stats.iterrows()}
    soil_min_max = {row['Band']: (row['Min'], row['Max']) for _, row in soil_stats.iterrows()}
    weather_min_max = {row['Band']: (row['Min'], row['Max']) for _, row in weather_stats.iterrows()}
    
    return s1_min_max, s2_min_max, modis_min_max, soil_min_max, weather_min_max

def normalize_modalities(modalities, s1_min_max, s2_min_max, modis_min_max, soil_min_max, weather_min_max):
    modality_names = ['Sentinel-1', 'Sentinel-2', 'MODIS', 'Soil', 'Weather']
    band_lists = [
        ['vv', 'vh'],
        ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'],
        ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7'],
        ['aws100', 'aws150', 'aws999', 'nccpi3all', 'nccpi3corn', 'rootznaws', 'soc150', 'soc999', 'pctearthmc'],
        ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
    ]
    stats = [s1_min_max, s2_min_max, modis_min_max, soil_min_max, weather_min_max]
    
    normalized_modalities = []
    for modality, name, bands, stat in zip(modalities, modality_names, band_lists, stats):
        if stat is None:
            # Fallback: normalize using min/max of the current batch
            vmin = modality.min(dim=(1, 2, 3), keepdim=True)[0]
            vmax = modality.max(dim=(1, 2, 3), keepdim=True)[0]
            modality = (modality - vmin) / (vmax - vmin + 1e-8)
        else:
            normalized_channels = []
            for i, band in enumerate(bands):
                vmin, vmax = stat.get(band, (modality[i].min(), modality[i].max()))
                channel = (modality[i] - vmin) / (vmax - vmin + 1e-8)
                normalized_channels.append(channel)
            modality = torch.stack(normalized_channels, dim=0)
        normalized_modalities.append(modality)
    return normalized_modalities


class MoELayer(nn.Module):
    def __init__(self, num_experts, expert_capacity, d_model, d_ff):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        gate_logits = self.gate(x)  # (batch_size, seq_len, num_experts)
        gate_probs = torch.nn.functional.softmax(gate_logits, dim=-1)
        _, top_k_indices = gate_probs.topk(1, dim=-1)  # (batch_size, seq_len, 1)
        
        # Create one-hot mask for top-1 expert
        mask = torch.nn.functional.one_hot(top_k_indices.squeeze(-1), num_classes=self.num_experts)  # (batch_size, seq_len, num_experts)
        mask = mask.to(x.dtype)  # Ensure mask is float for multiplication
        
        # Compute all expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)  # (num_experts, batch_size, seq_len, d_model)
        
        # Apply mask to select top-1 expert output
        # Reshape mask to (batch_size, seq_len, num_experts, 1) for broadcasting
        mask = mask.unsqueeze(-1)  # (batch_size, seq_len, num_experts, 1)
        # Transpose expert_outputs to (batch_size, seq_len, num_experts, d_model)
        expert_outputs = expert_outputs.permute(1, 2, 0, 3)
        # Element-wise multiply and sum over num_experts
        output = (expert_outputs * mask).sum(dim=2)  # (batch_size, seq_len, d_model)
        
        return output  # (batch_size, seq_len, d_model)

class MultiModalLIMoE(nn.Module):
    def __init__(self, num_layers, num_moe_layers, num_experts, d_model, d_ff, num_heads, modality_channels, num_classes):
        super().__init__()
        self.modality_embeds = nn.ModuleList([nn.Linear(channels, d_model) for channels in modality_channels])
        moe_indices = set(range(num_layers - num_moe_layers, num_layers))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=0.1,
                batch_first=True
            ) if i not in moe_indices else MoELayer(
                num_experts=num_experts,
                expert_capacity=128,
                d_model=d_model,
                d_ff=d_ff
            )
            for i in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(len(modality_channels) * d_model, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, modalities):
        modality_reprs = []
        for m, embed in zip(modalities, self.modality_embeds):
            # pdb.set_trace()
            m = m.permute(0, 2, 3, 1)  # (batch_size, patch_size, patch_size, channels)
            m = m.reshape(m.size(0), m.size(2)*m.size(2), -1)  # (batch_size, patch_size * patch_size, channels)
            m_emb = embed(m)  # (batch_size, patch_size * patch_size, d_model)
            for layer in self.layers:
                m_emb = layer(m_emb)
            pdb.set_trace()
            m_repr = m_emb.mean(dim=1)  # (batch_size, d_model)
            modality_reprs.append(m_repr)
        combined_repr = torch.cat(modality_reprs, dim=1)  # (batch_size, num_modalities * d_model)
        logits = self.classifier(combined_repr)
        return logits

class MultiModalLIMoELightning(pl.LightningModule):
    def __init__(self, variant='H/14', num_classes=256, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        modality_channels = [2, 12, 7, 9, 7]
        if variant == 'H/14':
            self.model = MultiModalLIMoE(
                num_layers=32,
                num_moe_layers=12,
                num_experts=32,
                d_model=1280,
                d_ff=5120,
                num_heads=16,
                modality_channels=modality_channels,
                num_classes=num_classes
            )
        elif variant == 'B/16':
            self.model = MultiModalLIMoE(
                num_layers=12,
                num_moe_layers=4,
                num_experts=8,
                d_model=768,
                d_ff=3072,
                num_heads=12,
                modality_channels=modality_channels,
                num_classes=num_classes
            )
        else:
            raise ValueError("Unsupported variant")
        self.criterion = nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, modalities):
        return self.model(modalities)

    def training_step(self, batch, batch_idx):
        modalities, labels = batch
        # pdb.set_trace()
        s1_min_max, s2_min_max, modis_min_max, soil_min_max, weather_min_max = load_statistics()
        modalities = normalize_modalities(modalities, s1_min_max, s2_min_max, modis_min_max, soil_min_max, weather_min_max)
        logits = self(modalities)
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        modalities, labels = batch
        s1_min_max, s2_min_max, modis_min_max, soil_min_max, weather_min_max = load_statistics()
        modalities = normalize_modalities(modalities, s1_min_max, s2_min_max, modis_min_max, soil_min_max, weather_min_max)
        logits = self(modalities)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.validation_step_outputs.append({'loss': loss, 'preds': preds.cpu(), 'labels': labels.cpu()})
        return {'loss': loss, 'preds': preds, 'labels': labels}

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        all_preds = torch.cat([x['preds'] for x in outputs]).numpy()
        all_labels = torch.cat([x['labels'] for x in outputs]).numpy()
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        micro_f1 = f1_score(all_labels, all_preds, average='micro')
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_macro_f1', macro_f1, prog_bar=True)
        self.log('val_micro_f1', micro_f1, prog_bar=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        modalities, labels = batch
        s1_min_max, s2_min_max, modis_min_max, soil_min_max, weather_min_max = load_statistics()
        modalities = normalize_modalities(modalities, s1_min_max, s2_min_max, modis_min_max, soil_min_max, weather_min_max)
        logits = self(modalities)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.test_step_outputs.append({'loss': loss, 'preds': preds.cpu(), 'labels': labels.cpu()})
        return {'loss': loss, 'preds': preds, 'labels': labels}

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        all_preds = torch.cat([x['preds'] for x in outputs]).numpy()
        all_labels = torch.cat([x['labels'] for x in outputs]).numpy()
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        micro_f1 = f1_score(all_labels, all_preds, average='micro')
        self.log('test_loss', avg_loss, prog_bar=True)
        self.log('test_macro_f1', macro_f1, prog_bar=True)
        self.log('test_micro_f1', micro_f1, prog_bar=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

class MultiModalDataModule(pl.LightningDataModule):
    def __init__(self, train_dirs, val_dirs, test_dirs, batch_size=32, num_workers=4):
        super().__init__()
        self.train_dirs = train_dirs
        self.val_dirs = val_dirs
        self.test_dirs = test_dirs
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = MultiModalDataset(**self.train_dirs)
        self.val_ds = MultiModalDataset(**self.val_dirs)
        self.test_ds = MultiModalDataset(**self.test_dirs)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)

if __name__ == "__main__":

    # Initialize data module
    # data_module = MultiModalDataModule(train_dirs, val_dirs, test_dirs, batch_size=32, num_workers=4)

    # Initialize model
    model = MultiModalLIMoELightning(variant='B/16', num_classes=256, learning_rate=1e-4)

    # Set up checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='limoe-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )
    num_workers = 4
    train_ds = MultiModalDataset(sentinel1_dir='/work/mech-ai-scratch/rtali/gis-sentinel1/final_s1',
    sentinel2_dir='/work/mech-ai-scratch/rtali/gis-sentinel2/final_s2_v3',
    modis_dir='/work/mech-ai-scratch/rtali/gis-modis/modis',
    crop_dir='/work/mech-ai-scratch/rtali/gis-CDL/final_CDL',
    soil_dir='/work/mech-ai-scratch/rtali/AI_READY_IOWA/SOIL/CRSchange',
    weather_dir='/work/mech-ai-scratch/rtali/AI_READY_IOWA/WEATHER_TIFFS',)
    train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=num_workers)

    for batch in train_dataloader:
        modalities, labels = batch
        for i, m in enumerate(modalities):
            print(f"Modality {i} batch shape: {m.shape}")
        print(f"Labels shape: {labels.shape}")
        break
    pdb.set_trace()
    # Train and test
    trainer.fit(model, train_dataloader)
    # trainer.test(model, data_module)