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