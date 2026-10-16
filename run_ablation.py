import os
import csv
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision.models import vgg16, VGG16_Weights
from pytorch_msssim import ms_ssim


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path("/home/ex5/Desktop/ARTEN_Project")

SPLIT_DIR = PROJECT_ROOT / "ARTEN_Ablation_Study" / "splits"
OUTPUT_ROOT = PROJECT_ROOT / "ARTEN_Ablation_Study" / "results"

INPUT_DIR = PROJECT_ROOT / "TrainingPairs" / "input"
TARGET_DIR = PROJECT_ROOT / "TrainingPairs" / "target"

IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5

SEED = 42
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class ARTENSplitDataset(Dataset):

    def __init__(self, split_csv):

        self.records = []

        with open(split_csv, "r", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                self.records.append(row)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        row = self.records[idx]

        input_path = row["input_path"]
        target_path = row["target_path"]

        inp = cv2.imread(input_path, cv2.IMREAD_COLOR)
        tgt = cv2.imread(target_path, cv2.IMREAD_COLOR)

        if inp is None:
            raise RuntimeError(f"Unable to read input image: {input_path}")

        if tgt is None:
            raise RuntimeError(f"Unable to read target image: {target_path}")

        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)

        inp = torch.from_numpy(inp).permute(2, 0, 1).float() / 255.0
        tgt = torch.from_numpy(tgt).permute(2, 0, 1).float() / 255.0

        return inp, tgt


# ============================================================
# SPATIAL ATTENTION
# ============================================================

class SpatialAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        x_cat = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        attention = self.sigmoid(
            self.conv(x_cat)
        )

        return attention


# ============================================================
# RESIDUAL / ATTENTION BLOCK
# ============================================================

class ResidualAttentionBlock(nn.Module):

    def __init__(
        self,
        channels,
        use_attention=True,
        use_residual=True
    ):

        super().__init__()

        self.use_attention = use_attention
        self.use_residual = use_residual

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1
        )

        if self.use_attention:
            self.attention = SpatialAttention()

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.use_attention:

            A = self.attention(out)
            out = out * A

        if self.use_residual:
            out = residual + out

        return out


# ============================================================
# ENCODER BLOCK
# ============================================================

class EncoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        use_attention=True,
        use_residual=True
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(),

            ResidualAttentionBlock(
                out_channels,
                use_attention=use_attention,
                use_residual=use_residual
            )
        )

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        features = self.block(x)

        pooled = self.pool(features)

        return features, pooled


# ============================================================
# DECODER BLOCK
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_attention=True,
        use_residual=True
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = nn.Sequential(

            nn.Conv2d(
                out_channels + skip_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(),

            ResidualAttentionBlock(
                out_channels,
                use_attention=use_attention,
                use_residual=use_residual
            )
        )

    def forward(self, x, skip):

        x = self.up(x)

        x = torch.cat(
            [x, skip],
            dim=1
        )

        x = self.conv(x)

        return x


# ============================================================
# MULTI-SCALE HEAD
# ============================================================

class MultiScaleHead(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.head = nn.Conv2d(
            channels,
            3,
            1
        )

    def forward(self, x):

        return torch.sigmoid(
            self.head(x)
        )


# ============================================================
# ARTEN MODEL
# ============================================================

class ARTEN(nn.Module):

    def __init__(
        self,
        use_attention=True,
        use_residual=True,
        use_multiscale=True
    ):

        super().__init__()

        self.use_multiscale = use_multiscale

        self.enc1 = EncoderBlock(
            3,
            64,
            use_attention,
            use_residual
        )

        self.enc2 = EncoderBlock(
            64,
            128,
            use_attention,
            use_residual
        )

        self.enc3 = EncoderBlock(
            128,
            256,
            use_attention,
            use_residual
        )

        self.bottleneck = nn.Sequential(

            nn.Conv2d(
                256,
                512,
                3,
                padding=1
            ),

            nn.ReLU(),

            ResidualAttentionBlock(
                512,
                use_attention,
                use_residual
            )
        )

        self.dec3 = DecoderBlock(
            512,
            256,
            256,
            use_attention,
            use_residual
        )

        self.dec2 = DecoderBlock(
            256,
            128,
            128,
            use_attention,
            use_residual
        )

        self.dec1 = DecoderBlock(
            128,
            64,
            64,
            use_attention,
            use_residual
        )

        self.final = nn.Conv2d(
            64,
            3,
            1
        )

        if self.use_multiscale:

            self.ms1 = MultiScaleHead(256)
            self.ms2 = MultiScaleHead(128)
            self.ms3 = MultiScaleHead(64)

    def forward(self, x):

        s1, p1 = self.enc1(x)

        s2, p2 = self.enc2(p1)

        s3, p3 = self.enc3(p2)

        bottleneck = self.bottleneck(p3)

        d3 = self.dec3(
            bottleneck,
            s3
        )

        d2 = self.dec2(
            d3,
            s2
        )

        d1 = self.dec1(
            d2,
            s1
        )

        out = torch.sigmoid(
            self.final(d1)
        )

        if self.use_multiscale:

            ms_out1 = self.ms1(d3)
            ms_out2 = self.ms2(d2)
            ms_out3 = self.ms3(d1)

            return (
                out,
                ms_out1,
                ms_out2,
                ms_out3
            )

        return out


# ============================================================
# PERCEPTUAL LOSS
# ============================================================

class PerceptualLoss(nn.Module):

    def __init__(self):

        super().__init__()

        vgg = vgg16(
            weights=VGG16_Weights.DEFAULT
        )

        self.features = nn.Sequential(
            *list(vgg.features)[:16]
        )

        for param in self.features.parameters():
            param.requires_grad = False

        self.features.eval()

    def forward(self, pred, target):

        pred_feat = self.features(pred)
        target_feat = self.features(target)

        return F.l1_loss(
            pred_feat,
            target_feat
        )


# ============================================================
# EDGE LOSS
# ============================================================

def edge_loss(pred, target):

    sobel_x = torch.tensor(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ],
        dtype=torch.float32,
        device=pred.device
    )

    sobel_x = sobel_x.view(
        1,
        1,
        3,
        3
    )

    pred_gray = torch.mean(
        pred,
        dim=1,
        keepdim=True
    )

    target_gray = torch.mean(
        target,
        dim=1,
        keepdim=True
    )

    pred_edge = F.conv2d(
        pred_gray,
        sobel_x,
        padding=1
    )

    target_edge = F.conv2d(
        target_gray,
        sobel_x,
        padding=1
    )

    return F.l1_loss(
        pred_edge,
        target_edge
    )


# ============================================================
# MS-SSIM LOSS
# ============================================================

def ms_ssim_loss(pred, target):

    return 1 - ms_ssim(
        pred,
        target,
        data_range=1,
        size_average=True
    )


# ============================================================
# MULTI-SCALE LOSS
# ============================================================

def multiscale_loss(
    ms1,
    ms2,
    ms3,
    target
):

    target_64 = F.interpolate(
        target,
        size=(64, 64)
    )

    target_128 = F.interpolate(
        target,
        size=(128, 128)
    )

    l1 = F.l1_loss(
        ms1,
        target_64
    )

    l2 = F.l1_loss(
        ms2,
        target_128
    )

    l3 = F.l1_loss(
        ms3,
        target
    )

    return l1 + l2 + l3


# ============================================================
# HYBRID LOSS
# ============================================================

def hybrid_loss(
    pred,
    ms1,
    ms2,
    ms3,
    target,
    perceptual_loss_fn
):

    l1 = F.l1_loss(
        pred,
        target
    )

    msssim = ms_ssim_loss(
        pred,
        target
    )

    perceptual = perceptual_loss_fn(
        pred,
        target
    )

    edge = edge_loss(
        pred,
        target
    )

    multi = multiscale_loss(
        ms1,
        ms2,
        ms3,
        target
    )

    total_loss = (
        1.0 * l1 +
        0.5 * msssim +
        0.1 * perceptual +
        0.1 * edge +
        0.2 * multi
    )

    return total_loss


# ============================================================
# LOSS WITHOUT MULTI-SCALE SUPERVISION
# ============================================================

def loss_without_multiscale(
    pred,
    target,
    perceptual_loss_fn
):

    l1 = F.l1_loss(
        pred,
        target
    )

    msssim = ms_ssim_loss(
        pred,
        target
    )

    perceptual = perceptual_loss_fn(
        pred,
        target
    )

    edge = edge_loss(
        pred,
        target
    )

    total_loss = (
        1.0 * l1 +
        0.5 * msssim +
        0.1 * perceptual +
        0.1 * edge
    )

    return total_loss


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(pred, target):

    mse = F.mse_loss(
        pred,
        target
    )

    if mse.item() == 0:
        return float("inf")

    return (
        10.0 *
        torch.log10(
            1.0 / mse
        )
    ).item()


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(pred, target):

    from pytorch_msssim import ssim

    return ssim(
        pred,
        target,
        data_range=1,
        size_average=True
    ).item()


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    use_multiscale
):

    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for inputs, targets in loader:

        inputs = inputs.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(inputs)

        if use_multiscale:

            pred, ms1, ms2, ms3 = outputs

        else:

            pred = outputs

        loss = F.l1_loss(
            pred,
            targets
        )

        batch_size = inputs.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_psnr += (
            calculate_psnr(
                pred,
                targets
            ) * batch_size
        )

        total_ssim += (
            calculate_ssim(
                pred,
                targets
            ) * batch_size
        )

        count += batch_size

    return (
        total_loss / count,
        total_psnr / count,
        total_ssim / count
    )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    model_name,
    use_attention,
    use_residual,
    use_multiscale
):

    print("\n")
    print("=" * 70)
    print(f"TRAINING MODEL: {model_name}")
    print("=" * 70)

    output_dir = OUTPUT_ROOT / model_name
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_csv = SPLIT_DIR / "train_split.csv"
    val_csv = SPLIT_DIR / "validation_split.csv"
    test_csv = SPLIT_DIR / "test_split.csv"

    train_dataset = ARTENSplitDataset(
        train_csv
    )

    val_dataset = ARTENSplitDataset(
        val_csv
    )

    test_dataset = ARTENSplitDataset(
        test_csv
    )

    print("Train samples :", len(train_dataset))
    print("Val samples   :", len(val_dataset))
    print("Test samples  :", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = ARTEN(
        use_attention=use_attention,
        use_residual=use_residual,
        use_multiscale=use_multiscale
    ).to(DEVICE)

    perceptual_loss_fn = PerceptualLoss().to(
        DEVICE
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE
    )

    best_val_loss = float("inf")

    history_path = output_dir / "training_history.csv"

    with open(history_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_psnr",
            "val_ssim",
            "learning_rate"
        ])

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"{model_name} Epoch {epoch + 1}/{EPOCHS}"
        )

        for inputs, targets in progress:

            inputs = inputs.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(inputs)

            if use_multiscale:

                pred, ms1, ms2, ms3 = outputs

                loss = hybrid_loss(
                    pred,
                    ms1,
                    ms2,
                    ms3,
                    targets,
                    perceptual_loss_fn
                )

            else:

                pred = outputs

                loss = loss_without_multiscale(
                    pred,
                    targets,
                    perceptual_loss_fn
                )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.5f}"
            )

        train_loss = (
            epoch_loss /
            len(train_loader)
        )

        val_loss, val_psnr, val_ssim = evaluate(
            model,
            val_loader,
            use_multiscale
        )

        scheduler.step(
            val_loss
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch + 1:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val L1: {val_loss:.6f} | "
            f"Val PSNR: {val_psnr:.4f} | "
            f"Val SSIM: {val_ssim:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        with open(
            history_path,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch + 1,
                train_loss,
                val_loss,
                val_psnr,
                val_ssim,
                current_lr
            ])

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "model_name": model_name
                },
                output_dir / "best_model.pth"
            )

            print(
                f"Best checkpoint saved: "
                f"{output_dir / 'best_model.pth'}"
            )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(
        output_dir / "best_model.pth",
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    test_l1, test_psnr, test_ssim = evaluate(
        model,
        test_loader,
        use_multiscale
    )

    print("\n")
    print("=" * 70)
    print(f"FINAL TEST RESULTS: {model_name}")
    print("=" * 70)
    print(f"Test L1   : {test_l1:.6f}")
    print(f"Test PSNR  : {test_psnr:.6f} dB")
    print(f"Test SSIM  : {test_ssim:.6f}")

    results_path = output_dir / "final_test_results.csv"

    with open(
        results_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "model",
            "test_l1",
            "test_psnr_db",
            "test_ssim"
        ])

        writer.writerow([
            model_name,
            test_l1,
            test_psnr,
            test_ssim
        ])

    print(
        f"Results saved to: {results_path}"
    )


# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

MODEL_CONFIGS = {

    "Full_ARTEN": {
        "use_attention": True,
        "use_residual": True,
        "use_multiscale": True
    },

    "ARTEN_Without_Spatial_Attention": {
        "use_attention": False,
        "use_residual": True,
        "use_multiscale": True
    },

    "ARTEN_Without_Residual_Connections": {
        "use_attention": True,
        "use_residual": False,
        "use_multiscale": True
    },

    "ARTEN_Without_MultiScale_Supervision": {
        "use_attention": True,
        "use_residual": True,
        "use_multiscale": False
    }
}


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys())
    )

    args = parser.parse_args()

    set_seed(SEED)

    config = MODEL_CONFIGS[
        args.model
    ]

    print("=" * 70)
    print("ARTEN CONTROLLED ABLATION STUDY")
    print("=" * 70)

    print("Model             :", args.model)
    print("Device            :", DEVICE)
    print("Image size        :", IMAGE_SIZE)
    print("Batch size        :", BATCH_SIZE)
    print("Epochs            :", EPOCHS)
    print("Learning rate     :", LEARNING_RATE)
    print("Seed              :", SEED)
    print("Attention         :", config["use_attention"])
    print("Residual          :", config["use_residual"])
    print("Multi-scale       :", config["use_multiscale"])

    if torch.cuda.is_available():

        print(
            "GPU               :",
            torch.cuda.get_device_name(0)
        )

    train_model(
        model_name=args.model,
        use_attention=config["use_attention"],
        use_residual=config["use_residual"],
        use_multiscale=config["use_multiscale"]
    )


if __name__ == "__main__":
    main()
