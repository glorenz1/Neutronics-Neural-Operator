import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


n_threads = int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads()))
torch.set_num_threads(n_threads)
torch.set_num_interop_threads(max(1, n_threads // 4))
print(f"PyTorch using {n_threads} threads")


MAX_CELLS = 85
N_IN = 3  # channels: enrich, coord_xn, coord_yn
MODES = 12
WIDTH = 32
DEPTH = 3
PAD = 4
LOG = 5


# center pad a 2D array with zeros to (target x target)
def pad_to(arr, target):
    h, w = arr.shape
    ph = (target - h) // 2
    pw = (target - w) // 2
    return np.pad(arr, ((ph, target - h - ph), (pw, target - w - pw)))


# load all sample_*.npz files from samples_dir into numpy arrays
def load_dataset(samples_dir, max_samples=None):
    pattern = os.path.join(samples_dir, "sample_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No sample_*.npz files found in {samples_dir}")
    if max_samples is not None:
        files = files[:max_samples]
    n_files = len(files)
    print(f"Loading {n_files} samples from {samples_dir}...")

    x = np.zeros((n_files, MAX_CELLS, MAX_CELLS, N_IN), dtype=np.float32)
    yp = np.zeros((n_files, MAX_CELLS, MAX_CELLS), dtype=np.float32)
    yk = np.zeros((n_files, 1), dtype=np.float32)
    mask = np.zeros((n_files, MAX_CELLS, MAX_CELLS), dtype=np.float32)

    failed = 0
    loaded = 0

    for fpath in files:
        try:
            sample = np.load(fpath)
            x[loaded, :, :, 0] = pad_to(sample["enrich"], MAX_CELLS)
            x[loaded, :, :, 1] = pad_to(sample["coord_xn"], MAX_CELLS)
            x[loaded, :, :, 2] = pad_to(sample["coord_yn"], MAX_CELLS)

            yp[loaded] = pad_to(sample["flux"], MAX_CELLS)
            mask[loaded] = pad_to(sample["mask"], MAX_CELLS)
            yk[loaded, 0] = float(sample["keff"])
            loaded += 1
        except Exception as e:
            print(f"  Warning: failed to load {fpath}: {e}")
            failed += 1

    if failed:
        print(f"  {failed} files failed; using {loaded} samples")
        x = x[:loaded]
        yp = yp[:loaded]
        yk = yk[:loaded]
        mask = mask[:loaded]

    print(f"  k_eff  range : [{yk.min():.4f}, {yk.max():.4f}]")
    print(f"  enrich range : [{x[:, :, :, 0].min():.4f}, {x[:, :, :, 0].max():.4f}]")
    print(f"Dataset loaded : {loaded} samples")
    return x, yp, yk, mask


class SpectralConv2d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        scale = 1 / (width * width)
        self.width = width
        self.modes = modes
        self.w1 = nn.Parameter(scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        xf = fft.rfft2(x)
        out = torch.zeros(B, self.width, xf.size(-2), xf.size(-1), dtype=torch.cfloat, device=x.device)

        # clip to available frequency bins in case grid is smaller than MODES
        mx = min(self.modes, xf.size(-2))
        my = min(self.modes, xf.size(-1))

        # apply learned weights to lower-left and upper-left frequency blocks
        out[:, :, :mx, :my] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :mx, :my], self.w1[:, :, :mx, :my])
        out[:, :, -mx:, :my] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -mx:, :my], self.w2[:, :, :mx, :my])
        return fft.irfft2(out, s=(H, W))


# 2D Fourier Neural Operator — predicts flux field (phi) and keff scalar from input channels
class FNO2d(nn.Module):
    def __init__(self, modes=MODES, width=WIDTH, depth=DEPTH, pad=PAD, n_in=N_IN):
        super().__init__()
        self.pad = pad
        self.fc0 = nn.Linear(n_in, width)
        self.spec = nn.ModuleList([SpectralConv2d(width, modes) for _ in range(depth)])
        self.pw = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])

        self.phi_head = nn.Sequential(
            nn.Linear(width, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.k_head = nn.Sequential(
            nn.Linear(width * 3, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # lift input channels to FNO feature width
        x = self.fc0(x).permute(0, 3, 1, 2)

        if self.pad:
            x = F.pad(x, [self.pad] * 4)

        for spec_conv, pw_conv in zip(self.spec, self.pw):
            x = F.gelu(spec_conv(x) + pw_conv(x))

        if self.pad:
            x = x[..., self.pad:-self.pad, self.pad:-self.pad]

        # predict flux field and normalize so it sums to unit L2 norm
        phi = self.phi_head(x.permute(0, 2, 3, 1)).squeeze(-1)
        phi = phi / (torch.norm(phi.reshape(phi.size(0), -1), dim=1).view(-1, 1, 1) + 1e-12)

        # global pool features for keff prediction, softplus keeps output positive
        k_mean = x.mean(dim=(-2, -1))
        k_max = x.amax(dim=(-2, -1))
        k_std = x.std(dim=(-2, -1))
        k = F.softplus(self.k_head(torch.cat([k_mean, k_max, k_std], dim=1))) + 1e-6
        return phi, k


# relative L2 loss over active (masked) cells only
def masked_rel_l2(pred, true, mask):
    pred_flat = (pred * mask).reshape(pred.size(0), -1)
    true_flat = (true * mask).reshape(true.size(0), -1)
    return torch.mean(torch.norm(pred_flat - true_flat, dim=1) / (torch.norm(true_flat, dim=1) + 1e-12))


# returns sample flux relative error, sample keff relative error, and keff MSE
def batch_metrics(pred_phi, true_phi, pred_k, true_k, mask):
    pred_flat = (pred_phi * mask).reshape(pred_phi.size(0), -1)
    true_flat = (true_phi * mask).reshape(true_phi.size(0), -1)

    flux_rel = torch.norm(pred_flat - true_flat, dim=1) / (torch.norm(true_flat, dim=1) + 1e-12)
    k_rel = torch.abs(pred_k - true_k).squeeze(1) / (torch.abs(true_k.squeeze(1)) + 1e-12)
    k_mse = ((pred_k - true_k) ** 2).mean()
    return flux_rel, k_rel, k_mse


# run model over a full dataloader and return averaged loss/error
def evaluate_model(model, loader):
    model.eval()
    flux_loss_sum = 0
    total_loss_sum = 0
    flux_err_sum = 0
    k_err_sum = 0
    n_seen = 0
    n_batches = 0

    with torch.no_grad():
        for x_batch, flux_batch, keff_batch, mask_batch in loader:
            pred_phi, pred_k = model(x_batch)
            loss_flux = masked_rel_l2(pred_phi, flux_batch, mask_batch)
            flux_rel, k_rel, k_mse = batch_metrics(pred_phi, flux_batch, pred_k, keff_batch, mask_batch)
            loss = loss_flux + 10 * k_mse
            flux_loss_sum += loss_flux.item()
            total_loss_sum += loss.item()
            flux_err_sum += flux_rel.sum().item()
            k_err_sum += k_rel.sum().item()
            n_seen += x_batch.size(0)
            n_batches += 1

    return {
        "flux_loss": flux_loss_sum / max(1, n_batches),
        "total_loss": total_loss_sum / max(1, n_batches),
        "flux_rel_err": flux_err_sum / max(1, n_seen),
        "k_rel_err": k_err_sum / max(1, n_seen),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to samples folder")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    samples_dir = args.data
    n_epochs = args.epochs
    batch_size = args.batch
    checkpoint_path = "fno_trained.pt"
    best_checkpoint_path = "fno_best.pt"

    val_frac = 0.15
    split_seed = 42

    x, yp, yk, masks = load_dataset(samples_dir)
    n_samples = len(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | {n_samples} samples | padded grid {MAX_CELLS}x{MAX_CELLS}")
    print(f"FNO config | modes={MODES} width={WIDTH} depth={DEPTH} pad={PAD} epochs={n_epochs}")

    # train/val split
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_samples)
    n_val = max(1, int(round(val_frac * n_samples))) if n_samples > 1 else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:] if n_val < n_samples else perm[:0]
    if len(train_idx) == 0:
        train_idx, val_idx = perm, perm[:0]

    # normalize inputs using train set statistics only
    x_mean = x[train_idx].mean(axis=(0, 1, 2))
    x_std = x[train_idx].std(axis=(0, 1, 2)) + 1e-8

    x_tensor = torch.tensor((x - x_mean) / x_std, dtype=torch.float32).to(device)
    flux_tensor = torch.tensor(yp, dtype=torch.float32).to(device)
    keff_tensor = torch.tensor(yk, dtype=torch.float32).to(device)
    mask_tensor = torch.tensor(masks, dtype=torch.float32).to(device)

    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

    train_ds = TensorDataset(x_tensor[train_idx_t], flux_tensor[train_idx_t], keff_tensor[train_idx_t], mask_tensor[train_idx_t])
    val_ds = TensorDataset(x_tensor[val_idx_t], flux_tensor[val_idx_t], keff_tensor[val_idx_t], mask_tensor[val_idx_t]) if len(val_idx) else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")) if val_ds is not None else None

    model = FNO2d(modes=MODES, width=WIDTH, depth=DEPTH, pad=PAD, n_in=N_IN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=5e-5)

    history = {
        "epoch": [],
        "train_flux_loss": [],
        "train_total_loss": [],
        "train_flux_rel_err": [],
        "train_k_rel_err": [],
        "val_flux_loss": [],
        "val_total_loss": [],
        "val_flux_rel_err": [],
        "val_k_rel_err": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 4
    min_delta = 1e-4
    null_epochs = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_flux = 0
        running_total = 0
        running_flux_rel = 0
        running_k_rel = 0
        n_seen = 0
        n_batches = 0

        for x_batch, flux_batch, keff_batch, mask_batch in train_loader:
            opt.zero_grad()
            pred_phi, pred_k = model(x_batch)
            loss_flux = masked_rel_l2(pred_phi, flux_batch, mask_batch)
            flux_rel, k_rel, k_mse = batch_metrics(pred_phi, flux_batch, pred_k, keff_batch, mask_batch)
            # flux relative L2 + weighted keff MSE
            loss = loss_flux + 3 * k_mse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # gradient clipping
            opt.step()

            running_flux += loss_flux.item()
            running_total += loss.item()
            running_flux_rel += flux_rel.sum().item()
            running_k_rel += k_rel.sum().item()
            n_seen += x_batch.size(0)
            n_batches += 1

        sched.step()

        if epoch % LOG == 0 or epoch == 1 or epoch == n_epochs:
            train_stats = {
                "flux_loss": running_flux / max(1, n_batches),
                "total_loss": running_total / max(1, n_batches),
                "flux_rel_err": running_flux_rel / max(1, n_seen),
                "k_rel_err": running_k_rel / max(1, n_seen),
            }
            val_stats = evaluate_model(model, val_loader) if val_loader is not None else None

            history["epoch"].append(epoch)
            history["train_flux_loss"].append(train_stats["flux_loss"])
            history["train_total_loss"].append(train_stats["total_loss"])
            history["train_flux_rel_err"].append(train_stats["flux_rel_err"])
            history["train_k_rel_err"].append(train_stats["k_rel_err"])

            if val_stats is not None:
                history["val_flux_loss"].append(val_stats["flux_loss"])
                history["val_total_loss"].append(val_stats["total_loss"])
                history["val_flux_rel_err"].append(val_stats["flux_rel_err"])
                history["val_k_rel_err"].append(val_stats["k_rel_err"])
                print(
                    f"Epoch {epoch:3d} | train flux={train_stats['flux_loss']:.4f} | val flux={val_stats['flux_loss']:.4f} | "
                    f"train krel={train_stats['k_rel_err']:.4e} | val krel={val_stats['k_rel_err']:.4e} | LR={sched.get_last_lr()[0]:.2e}"
                )
                if val_stats["total_loss"] < best_val_loss - min_delta:
                    best_val_loss = val_stats["total_loss"]
                    best_epoch = epoch
                    null_epochs = 0

                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "x_mean": x_mean,
                            "x_std": x_std,
                            "n_in": N_IN,
                            "max_cells": MAX_CELLS,
                            "train_idx": train_idx,
                            "val_idx": val_idx,
                            "fno_modes": MODES,
                            "fno_width": WIDTH,
                            "fno_depth": DEPTH,
                            "fno_pad": PAD,
                            "best_epoch": epoch,
                            "best_val_loss": best_val_loss,
                            "history": history,
                            "samples_dir": samples_dir,
                        },
                        best_checkpoint_path,
                    )
                    print(f"  -> New best val loss {best_val_loss:.4f} at epoch {epoch}; saved to {best_checkpoint_path}")
                else:
                    null_epochs += 1
                    print(f"  -> No val improvement for {null_epochs}/{patience} checks")

                if null_epochs >= patience:
                    print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch} with val loss {best_val_loss:.4f}")
                    break

            else:
                history["val_flux_loss"].append(None)
                history["val_total_loss"].append(None)
                history["val_flux_rel_err"].append(None)
                history["val_k_rel_err"].append(None)
                print(
                    f"Epoch {epoch:3d} | train flux={train_stats['flux_loss']:.4f} | train krel={train_stats['k_rel_err']:.4e} | "
                    f"LR={sched.get_last_lr()[0]:.2e}"
                )

    torch.save(
        {
            "model_state": model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "n_in": N_IN,
            "max_cells": MAX_CELLS,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "fno_modes": MODES,
            "fno_width": WIDTH,
            "fno_depth": DEPTH,
            "fno_pad": PAD,
            "epochs": n_epochs,
            "history": history,
            "samples_dir": samples_dir,
        },
        checkpoint_path,
    )
    print(f"Model saved to {checkpoint_path}")

    if os.path.exists(best_checkpoint_path):
        best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state"])
        print(f"Loaded best model from epoch {best_ckpt['best_epoch']}")

    train_stats = evaluate_model(model, train_loader)
    print(f"\nTrain relative flux error : {train_stats['flux_rel_err']:.4f}")
    print(f"Train relative keff error : {train_stats['k_rel_err']:.4f}")

    if val_loader is not None:
        val_stats = evaluate_model(model, val_loader)
        print(f"Val   relative flux error : {val_stats['flux_rel_err']:.4f}")
        print(f"Val   relative keff error : {val_stats['k_rel_err']:.4f}")
