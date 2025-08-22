# /home/idec2/high_resolution/cars_memristor_train.py
# Train the same snap-only memristor-aware model on Stanford Cars Dataset
# Dataset: stanford-car-dataset-by-classes-folder (ImageFolder format)

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# LUT utilities
# =========================================================
def load_memristor_values(pot_path: str, dep_path: str = None):
    def _read_txt(p):
        with open(p, "r") as f:
            vals = [float(x.strip()) for x in f if x.strip()]
        return torch.tensor(vals, dtype=torch.float32)

    if not os.path.exists(pot_path):
        raise FileNotFoundError(f"Potentiation file not found: {pot_path}")
    pot_vals = _read_txt(pot_path)

    dep_vals = None
    if dep_path is not None and os.path.exists(dep_path):
        dep_vals = _read_txt(dep_path)

    return pot_vals, dep_vals


def build_states_from_lut(pot_vals: torch.Tensor,
                          dep_vals: torch.Tensor = None,
                          zero_center: bool = True) -> torch.Tensor:
    if dep_vals is not None:
        states_neg = -torch.flip(dep_vals, dims=[0])
    else:
        states_neg = -torch.flip(pot_vals, dims=[0])

    states_pos = pot_vals.clone()
    states = torch.cat([states_neg, states_pos], dim=0)

    states = torch.unique(states)
    states, _ = torch.sort(states)

    if zero_center:
        states = states - states.mean()

    return states


# =========================================================
# Quantizer
# =========================================================
class MemristorQuantizerFromTensor:
    def __init__(self, states: torch.Tensor, device='cpu'):
        states = torch.unique(states).to(torch.float32).to(device)
        states, _ = torch.sort(states)
        self.states = states
        self.S = self.states.numel()

    def snap_to_state(self, w_fp32: torch.Tensor):
        w = w_fp32.unsqueeze(-1)
        d = torch.abs(w - self.states)
        idx = torch.argmin(d, dim=-1)
        w_snapped = self.states[idx]
        return w_snapped, idx

    def indices_to_weight(self, idx: torch.Tensor):
        return self.states[idx]


# =========================================================
# Device-aware Linear
# =========================================================
class DeviceAwareLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, quantizer=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.quantizer = quantizer

        self.weight_fp32 = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.uniform_(self.weight_fp32, a=-0.02, b=0.02)

        if bias:
            self.bias_fp32 = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_fp32', None)

        self.register_buffer('weight_idx', None)

    def _ensure_indices(self):
        if self.weight_idx is None:
            with torch.no_grad():
                w_snap, idx = self.quantizer.snap_to_state(self.weight_fp32)
                self.weight_fp32.copy_(w_snap)
                self.weight_idx = idx

    def forward(self, x):
        self._ensure_indices()
        w_device = self.quantizer.indices_to_weight(self.weight_idx)
        w = w_device + (self.weight_fp32 - self.weight_fp32.detach())
        b = self.bias_fp32 if self.use_bias and self.bias_fp32 is not None else None
        return torch.nn.functional.linear(x, w, b)

    @torch.no_grad()
    def project_after_step(self):
        self._ensure_indices()
        w_snapped, idx = self.quantizer.snap_to_state(self.weight_fp32)
        self.weight_fp32.copy_(w_snapped)
        self.weight_idx = idx


# =========================================================
# CNN (now 3-channel input for cars)
# =========================================================
class BaselineCNN_Device(nn.Module):
    def __init__(self, quantizer, image_size=128, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (image_size // 4) * (image_size // 4), 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = DeviceAwareLinear(128, num_classes, bias=True, quantizer=quantizer)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# =========================================================
# Train / Eval
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device, log_every=50):
    model.train()
    correct = total = 0
    running_loss = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for i, (x, y) in enumerate(pbar, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )

        optimizer.step()
        model.fc2.project_after_step()

        running_loss += float(loss.item()) * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        if i % log_every == 0:
            pbar.set_postfix(loss=running_loss/total,
                             acc=100.0*correct/total)
    return running_loss/total, 100.0*correct/total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct = total = 0
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        running_loss += float(loss.item()) * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss/total, 100.0*correct/total


# =========================================================
# Run for one device profile
# =========================================================
def run_for_device(device_name: str,
                   pot_path: str,
                   dep_path: str,
                   train_dir: str,
                   test_dir: str,
                   image_size: int,
                   batch_size: int,
                   epochs: int,
                   lr: float,
                   device: torch.device):

    print(f"\n===== Device profile: {device_name} =====")
    pot_vals, dep_vals = load_memristor_values(pot_path, dep_path)

    states = build_states_from_lut(pot_vals, dep_vals, zero_center=True)
    quant = MemristorQuantizerFromTensor(states, device=device)

    # transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])

    # Dataset from ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(test_dir,  transform=val_transform)
    classes = train_dataset.classes
    print(f"[resolver] Found {len(classes)} classes")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=torch.cuda.is_available(), num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              pin_memory=torch.cuda.is_available(), num_workers=2)

    model = BaselineCNN_Device(quantizer=quant, image_size=image_size, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    val_acc_per_epoch = []

    for epoch in range(1, epochs + 1):
        print(f"\n=== {device_name} Epoch {epoch}/{epochs} ===")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        val_acc_per_epoch.append(va_acc)
        print(f"[Train] loss={tr_loss:.4f}, acc={tr_acc:.2f}% | [Val] loss={va_loss:.4f}, acc={va_acc:.2f}%")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), f"best_cars_memristor_{device_name}.pth")
            print("==> Best model saved")

    print(f"Best Val Acc for {device_name}: {best_acc:.2f}%")
    return best_acc, val_acc_per_epoch


# =========================================================
# main
# =========================================================
def main():
    set_seed(42)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ROOT = "/home/idec2/.cache/kagglehub/datasets/jutrera/stanford-car-dataset-by-classes-folder/versions/2/car_data/car_data"
    TRAIN_DIR = os.path.join(ROOT, "train")
    TEST_DIR  = os.path.join(ROOT, "test")

    image_size = 128
    batch_size = 16
    epochs = 30
    lr = 1e-3

    LUT_ROOT = "/home/idec2/brain_mri/value2"
    paths = {
        "C1D1": {
            "pot": os.path.join(LUT_ROOT, "C1D1_potentiation.txt"),
            "dep": os.path.join(LUT_ROOT, "C1D1_depression.txt"),
        },
        "C4D1": {
            "pot": os.path.join(LUT_ROOT, "C4D1_potentiation.txt"),
            "dep": os.path.join(LUT_ROOT, "C4D1_depression.txt"),
        },
    }

    for name, p in paths.items():
        try:
            acc, curve = run_for_device(
                device_name=name,
                pot_path=p["pot"],
                dep_path=p["dep"],
                train_dir=TRAIN_DIR,
                test_dir=TEST_DIR,
                image_size=image_size,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                device=torch_device
            )
        except Exception as e:
            print(f"[Error {name}] {e}")


if __name__ == "__main__":
    main()

