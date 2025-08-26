# /home/ty/IDEC/stanfordcar_resnet/stanfordcar/test2_fixed_full.py
# Stanford Cars (ImageFolder) + pretrained ResNet18 backbone
# Final classifier is DeviceAwareLinear (memristor-aware)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------- TorchVision compatibility (weights + mean/std) ----------------
try:
    WEIGHTS = models.ResNet18_Weights.IMAGENET1K_V1
    IMAGENET_MEAN = WEIGHTS.meta.get("mean", [0.485, 0.456, 0.406])
    IMAGENET_STD  = WEIGHTS.meta.get("std",  [0.229, 0.224, 0.225])
    def load_resnet18():
        return models.resnet18(weights=WEIGHTS)
except Exception:
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    def load_resnet18():
        return models.resnet18(pretrained=True)

# ---------------- Reproducibility ----------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- LUT utilities ----------------
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

# ---------------- Quantizer ----------------
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

# ---------------- Device-aware Linear ----------------
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
        # STE
        w = w_device + (self.weight_fp32 - self.weight_fp32.detach())
        b = self.bias_fp32 if self.use_bias and self.bias_fp32 is not None else None
        return torch.nn.functional.linear(x, w, b)

    @torch.no_grad()
    def project_after_step(self):
        self._ensure_indices()
        w_snapped, idx = self.quantizer.snap_to_state(self.weight_fp32)
        self.weight_fp32.copy_(w_snapped)
        self.weight_idx = idx

# ---------------- Model: ResNet18 backbone + memristor head ----------------
class ResNet18_MemristorHead(nn.Module):
    def __init__(self, quantizer, num_classes=196, freeze_backbone=True):
        super().__init__()
        self.backbone = load_resnet18()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc1 = nn.Linear(in_features, 256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = DeviceAwareLinear(256, num_classes, bias=True, quantizer=quantizer)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # unfreeze last block for slight adaptation
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True

    def forward(self, x):
        f = self.backbone(x)   # [B, 512]
        f = self.fc1(f)        # [B, 256]
        f = self.relu(f)
        f = self.dropout(f)
        out = self.fc2(f)      # [B, C]
        return out

# ---------------- Train / Eval ----------------
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
        # project memristor head
        model.fc2.project_after_step()

        running_loss += float(loss.item()) * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        if i % log_every == 0:
            pbar.set_postfix(loss=running_loss/total, acc=100.0*correct/total)

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

# ---------------- One device profile ----------------
def run_for_device(device_name: str,
                   pot_path: str,
                   dep_path: str,
                   train_dir: str,
                   test_dir: str,
                   batch_size: int,
                   epochs: int,
                   lr: float,
                   device: torch.device,
                   freeze_backbone: bool = True):

    print(f"\n===== Device profile: {device_name} =====")
    pot_vals, dep_vals = load_memristor_values(pot_path, dep_path)
    states = build_states_from_lut(pot_vals, dep_vals, zero_center=True)
    quant = MemristorQuantizerFromTensor(states, device=device)

    image_size = 224
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size*1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    val_set   = datasets.ImageFolder(test_dir,  transform=val_tf)
    num_classes = len(train_set.classes)
    print(f"[resolver] classes={num_classes}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=torch.cuda.is_available())

    model = ResNet18_MemristorHead(quantizer=quant, num_classes=num_classes,
                                   freeze_backbone=freeze_backbone).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(optim_params, lr=lr, weight_decay=1e-4)

    best_acc = 0.0
    best_path = f"best_cars_memristor_resnet18_{device_name}.pth"
    val_curve = []

    for epoch in range(1, epochs + 1):
        print(f"\n=== {device_name} Epoch {epoch}/{epochs} ===")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        val_curve.append(va_acc)
        print(f"[Train] loss={tr_loss:.4f}, acc={tr_acc:.2f}% | [Val] loss={va_loss:.4f}, acc={va_acc:.2f}%")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)
            print(f"==> Best saved: {best_path}")

    print(f"Best Val Acc ({device_name}): {best_acc:.2f}%")
    return best_acc, best_path, val_curve

# ---------------- main ----------------
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Adjust ROOT/LUT_ROOT if your username or cache path differs
    ROOT = "/home/ty/.cache/kagglehub/datasets/jutrera/stanford-car-dataset-by-classes-folder/versions/2/car_data/car_data"
    TRAIN_DIR = os.path.join(ROOT, "train")
    TEST_DIR  = os.path.join(ROOT, "test")

    batch_size = 32        # reduce to 16 if OOM on 2GB VRAM
    epochs = 20            # start modest, then increase
    lr = 1e-3
    FREEZE_BACKBONE = True

    LUT_ROOT = "/home/ty/IDEC/brain_mri/value2"
    paths = {
        "C1D1": {
            "pot": os.path.join(LUT_ROOT, "C1D1_potentiation.txt"),
            "dep": os.path.join(LUT_ROOT, "C1D1_depression.txt"),
        },
        "C4D1": {
            "pot": os.path.join(LUT_ROOT, "C4D1_potentiation.txt"),
            "dep": os.path.join(LUT_ROOT, "C4D1_depression.txt"),
        },
        "C1D8": {
            "pot": os.path.join(LUT_ROOT, "C1D8_potentiation.txt"),
            "dep": os.path.join(LUT_ROOT, "C1D8_depression.txt"),
        },
        "Identical spikes": {
            "pot": os.path.join(LUT_ROOT, "Identical spikes_potentiation.txt"),
            "dep": os.path.join(LUT_ROOT, "Identical spikes_depression.txt"),
        },
        "Non spikes_1": {
            "pot": os.path.join(LUT_ROOT, "Non spikes_1_potentiation.txt"),
            "dep": os.path.join(LUT_ROOT, "Non spikes_1_depression.txt"),
        },
        "Non spikes_2": {
            "pot": os.path.join(LUT_ROOT, "Non spikes_2_potentiation.txt"),  # fixed
            "dep": os.path.join(LUT_ROOT, "Non spikes_2_depression.txt"),    # fixed
        },
    }

    results = {}
    acc_curves = {}

    for name, p in paths.items():
        try:
            acc, ckpt, curve = run_for_device(
                device_name=name,
                pot_path=p["pot"],
                dep_path=p["dep"],
                train_dir=TRAIN_DIR,
                test_dir=TEST_DIR,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                device=device,
                freeze_backbone=FREEZE_BACKBONE
            )
            results[name] = {"best_val_acc": acc, "ckpt": ckpt}
            acc_curves[name] = curve
        except Exception as e:
            print(f"[Error {name}] {e}")

    print("\n===== Summary =====")
    for k, v in results.items():
        curve = acc_curves.get(k, [])
        mean_acc = np.mean(curve) if len(curve) > 0 else 0.0
        print(f"{k}: best_val_acc={v['best_val_acc']:.2f}%, mean_val_acc={mean_acc:.2f}% ckpt={v['ckpt']}")

    if len(acc_curves) > 0:
        plt.figure(figsize=(9, 6))
        for name, curve in acc_curves.items():
            if len(curve) > 0:
                plt.plot(range(1, len(curve) + 1), curve, marker='o', label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy (%)")
        plt.title("Validation Accuracy per Epoch for Each Device (ResNet18 head)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("device_accuracy_comparison_resnet18.png", dpi=150)
        try:
            plt.show()
        except Exception:
            pass
    else:
        print("No curves to plot.")

if __name__ == "__main__":
    main()
