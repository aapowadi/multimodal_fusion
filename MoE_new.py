import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.optimization import Adafactor
from torch.optim.lr_scheduler import LambdaLR
import math

# Configuration
class Config:
    num_modalities = 3
    channels = [19, 29, 24]
    patch_size = 16
    image_size = 16
    num_experts = 32
    experts_per_layer = 1  # Top-1 routing (K=1)
    hidden_size = 512
    num_layers = 12
    num_heads = 8
    ffn_dim = 2048
    sequence_length = (image_size // patch_size) ** 2  # e.g., 196 for 224x224 with 16x16 patches
    output_dim = 512
    dropout = 0.1
    learning_rate = 1e-3
    weight_decay = 1e-5
    warmup_steps = 10000
    total_steps = 100000
    temperature_init = 10.0
    batch_size = 32
    entropy_threshold = math.log(4)  # Soft minimum of 4 experts
    aux_loss_weight = 0.01

# MLP Expert for MoE
class MLPExpert(nn.Module):
    def __init__(self, hidden_size, ffn_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)

# MoE Layer with Entropy Losses
class MoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, experts_per_layer, ffn_dim, dropout, num_modalities):
        super().__init__()
        self.num_experts = num_experts
        self.experts_per_layer = experts_per_layer
        self.experts = nn.ModuleList([MLPExpert(hidden_size, ffn_dim, dropout) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_size, num_experts)
        self.num_modalities = num_modalities

    def forward(self, x, modality_mask):
        batch_size, seq_len, _ = x.shape
        gate_logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Batch Priority Routing (BPR): Prioritize tokens by max probability
        max_probs = gate_probs.max(dim=-1)[0]  # [batch_size, seq_len]
        sorted_indices = max_probs.view(-1).argsort(descending=True)
        sorted_indices = sorted_indices[:int(self.num_experts * seq_len * 1.2)]  # 1.2x capacity
        sorted_probs = gate_probs.view(-1, self.num_experts)[sorted_indices]
        sorted_inputs = x.view(-1, x.size(-1))[sorted_indices]
        sorted_mask = modality_mask.view(-1, self.num_modalities)[sorted_indices]

        # Process tokens with top-k experts
        top_k_probs, top_k_indices = sorted_probs.topk(self.experts_per_layer, dim=-1)
        top_k_probs = F.softmax(top_k_probs, dim=-1)
        expert_output = torch.zeros_like(sorted_inputs, dtype=torch.bfloat16)
        for i in range(self.experts_per_layer):
            expert_idx = top_k_indices[:, i]
            prob = top_k_probs[:, i].unsqueeze(-1)
            for j in range(self.num_experts):
                mask = (expert_idx == j).float().unsqueeze(-1)
                if mask.sum() > 0:
                    expert_input = sorted_inputs * mask
                    expert_output_j = self.experts[j](expert_input)
                    expert_output += prob * expert_output_j * mask

        # Reshape output back to original shape
        output = torch.zeros(batch_size * seq_len, x.size(-1), dtype=torch.bfloat16, device=x.device)
        output[sorted_indices] = expert_output
        output = output.view(batch_size, seq_len, x.size(-1))

        # Compute auxiliary losses (local and global entropy)
        aux_losses = []
        for m in range(self.num_modalities):
            modality_indices = modality_mask[:, :, m].bool().view(batch_size, seq_len)
            if modality_indices.sum() > 0:
                modality_probs = gate_probs[modality_indices]  # [n_m, num_experts]
                # Local entropy loss
                local_entropy = -(modality_probs * torch.log(modality_probs + 1e-10)).sum(dim=-1).mean()
                # Global entropy loss
                marginal_probs = modality_probs.mean(dim=0)
                global_entropy = -(marginal_probs * torch.log(marginal_probs + 1e-10)).sum()
                global_entropy_loss = torch.relu(Config.entropy_threshold + global_entropy)
                aux_losses.append(local_entropy + global_entropy_loss)
        aux_loss = sum(aux_losses) / len(aux_losses) if aux_losses else torch.tensor(0.0).to(x.device)

        return output, aux_loss

# Transformer Layer
class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

# LIMoE Model
class LIMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Per-modality patch embeddings
        self.patch_embeds = nn.ModuleList([
            nn.Conv2d(channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
            for channels in config.channels
        ])
        self.pos_encoding = self.create_positional_encoding(config.sequence_length, config.hidden_size)
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            if (i + 1) % 2 == 0:
                self.layers.append(
                    MoELayer(
                        config.hidden_size,
                        config.num_experts,
                        config.experts_per_layer,
                        config.ffn_dim,
                        config.dropout,
                        config.num_modalities
                    )
                )
            else:
                self.layers.append(TransformerLayer(config.hidden_size, config.num_heads, config.ffn_dim, config.dropout))
        self.norm = nn.LayerNorm(config.hidden_size)
        self.projection = nn.ModuleList([nn.Linear(config.hidden_size, config.output_dim) for _ in range(config.num_modalities)])
        self.temperature = nn.Parameter(torch.tensor(config.temperature_init))

    def create_positional_encoding(self, seq_len, hidden_size):
        pe = torch.zeros(seq_len, hidden_size)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.to(torch.bfloat16)

    def forward(self, inputs, modality_idx):
        # inputs: [batch_size, channels, height, width]
        batch_size = inputs.size(0)
        x = self.patch_embeds[modality_idx](inputs)  # [batch_size, hidden_size, h/p, w/p]
        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.config.hidden_size)  # [batch_size, seq_len, hidden_size]
        x = x + self.pos_encoding.to(x.device)

        # Modality mask
        modality_mask = torch.zeros(batch_size, self.config.sequence_length, self.config.num_modalities, device=x.device)
        modality_mask[:, :, modality_idx] = 1.0

        # Pass through layers
        aux_loss = 0.0
        for layer in self.layers:
            if isinstance(layer, MoELayer):
                x, layer_aux_loss = layer(x, modality_mask)
                aux_loss += layer_aux_loss
            else:
                x = layer(x)

        # Final representation
        x = self.norm(x.mean(dim=1))  # [batch_size, hidden_size]
        z = self.projection[modality_idx](x).to(torch.bfloat16)  # [batch_size, output_dim]
        return z, aux_loss

# PyTorch Lightning Module
class LIMoELightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = LIMoE(config)

    def forward(self, inputs, modality_idx):
        return self.model(inputs, modality_idx)

    def training_step(self, batch, batch_idx):
        images, labels = batch  # images: list of [batch_size, channels_i, height, width]
        batch_size = images[0].size(0)
        loss = 0.0
        aux_loss = 0.0
        representations = []
        
        # Process each modality
        for m in range(self.config.num_modalities):
            input_m = images[m]  # [batch_size, channels[m], height, width]
            z_m, aux_m = self(input_m, m)
            representations.append(z_m)
            aux_loss += aux_m

        # Contrastive loss (InfoNCE) for all modality pairs
        for m1 in range(self.config.num_modalities):
            for m2 in range(m1 + 1, self.config.num_modalities):
                z1, z2 = representations[m1], representations[m2]
                logits = torch.matmul(z1, z2.T) / self.model.temperature
                labels = torch.arange(batch_size, device=logits.device)
                loss += (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        loss = loss / (self.config.num_modalities * (self.config.num_modalities - 1) / 2)
        total_loss = loss + self.config.aux_loss_weight * aux_loss
        self.log("train_loss", loss, prog_bar=True)
        self.log("aux_loss", aux_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch  # images: list of [batch_size, channels_i, height, width]
        batch_size = images[0].size(0)
        loss = 0.0
        representations = []
        
        for m in range(self.config.num_modalities):
            input_m = images[m]  # [batch_size, channels[m], height, width]
            z_m, _ = self(input_m, m)
            representations.append(z_m)

        # Contrastive loss (InfoNCE) for all modality pairs
        for m1 in range(self.config.num_modalities):
            for m2 in range(m1 + 1, self.config.num_modalities):
                z1, z2 = representations[m1], representations[m2]
                logits = torch.matmul(z1, z2.T) / self.model.temperature
                labels = torch.arange(batch_size, device=logits.device)
                loss += (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        loss = loss / (self.config.num_modalities * (self.config.num_modalities - 1) / 2)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adafactor(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False
        )

        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            progress = (step - self.config.warmup_steps) / (self.config.total_steps - self.config.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

# Synthetic Data Module
class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        # Generate synthetic data with correct channel counts for each modality
        self.train_dataset = [
            (
                [
                    # Normalize to [-1, 1] as per LIMoE paper
                    2 * torch.randn(self.config.batch_size, channels, self.config.image_size, self.config.image_size) - 1
                    for channels in self.config.channels
                ],  # List of [batch_size, channels_i, height, width]
                torch.randint(0, 10, (self.config.batch_size,))
            )
            for _ in range(1000)
        ]
        self.val_dataset = [
            (
                [
                    2 * torch.randn(self.config.batch_size, channels, self.config.image_size, self.config.image_size) - 1
                    for channels in self.config.channels
                ],
                torch.randint(0, 10, (self.config.batch_size,))
            )
            for _ in range(100)
        ]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=None, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=None)

# Main Training Script
if __name__ == "__main__":
    config = Config()
    model = LIMoELightning(config)
    data_module = SyntheticDataModule(config)

    trainer = pl.Trainer(
        max_steps=config.total_steps,
        accelerator="auto",
        precision="bf16-mixed",
        devices="auto",
        strategy="auto",
        log_every_n_steps=10
    )

    trainer.fit(model, data_module)