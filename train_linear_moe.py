# train_linear_moe.py
"""
Training script that first fits a simple linear regression model on synthetic data,
then builds a Mixture-of-Experts (MoE) model with multiple linear experts and a gating
network. The script logs training loss to `training_log.txt`.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Synthetic dataset
N_SAMPLES = 1000
INPUT_DIM = 20
X = np.random.randn(N_SAMPLES, INPUT_DIM).astype(np.float32)
true_w = np.random.randn(INPUT_DIM, 1).astype(np.float32)
y = X @ true_w + 0.1 * np.random.randn(N_SAMPLES, 1).astype(np.float32)

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ---------- Linear Regression Model ----------
class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc(x)

linear = LinearModel(INPUT_DIM)
criterion = nn.MSELoss()
optimizer = optim.Adam(linear.parameters(), lr=0.01)

log_lines = []
log_lines.append("--- Linear Regression Training ---\n")
for epoch in range(10):
    epoch_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = linear(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(dataset)
    log_lines.append(f"Epoch {epoch+1:02d}, Loss: {epoch_loss:.6f}\n")

# ---------- Mixture of Experts Model ----------
class MoE(nn.Module):
    def __init__(self, in_dim, num_experts=4, out_dim=1):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(in_dim, num_experts)
    def forward(self, x):
        gate_logits = self.gate(x)
        gate_weights = torch.softmax(gate_logits, dim=1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        out = torch.sum(gate_weights.unsqueeze(1) * expert_outputs, dim=2)
        return out

moe = MoE(INPUT_DIM, num_experts=4)
criterion_moe = nn.MSELoss()
optimizer_moe = optim.Adam(moe.parameters(), lr=0.01)

log_lines.append("\n--- MoE Training ---\n")
for epoch in range(15):
    epoch_loss = 0.0
    for xb, yb in loader:
        optimizer_moe.zero_grad()
        preds = moe(xb)
        loss = criterion_moe(preds, yb)
        loss.backward()
        optimizer_moe.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(dataset)
    log_lines.append(f"MoE Epoch {epoch+1:02d}, Loss: {epoch_loss:.6f}\n")

# Save log to file
with open("training_log.txt", "w") as f:
    f.writelines(log_lines)

print("Training complete. Log saved to training_log.txt")
