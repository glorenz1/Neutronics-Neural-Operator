import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle


MAX_CELLS = 85
N_IN = 3
DEFAULT_OUT_DIR = "fno_results"


def load_checkpoint(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError as exc:
        # Training checkpoints saved in this workflow include NumPy arrays and index lists.
        message = str(exc)
        if "Weights only load failed" not in message:
            raise
        print(
            "Checkpoint load hit PyTorch weights_only=True default; "
            "retrying with weights_only=False for this trusted local file."
        )
        return torch.load(checkpoint_path, map_location=device, weights_only=False)


def pad(arr, target):
    h, w = arr.shape
    ph = (target - h) // 2
    pw = (target - w) // 2
    return np.pad(arr, ((ph, target - h - ph), (pw, target - w - pw)))


def crop(arr, mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        return arr
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return arr[y0:y1 + 1, x0:x1 + 1]


def load(samples_dir, max_samples=None):
    pattern = os.path.join(samples_dir, "sample_*.npz")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No sample_*.npz files found in {samples_dir}")
    if max_samples is not None:
        files = files[:max_samples]

    n = len(files)
    print(f"Loading {n} OpenMC samples from {samples_dir}...")

    x = np.zeros((n, MAX_CELLS, MAX_CELLS, N_IN), dtype=np.float32)
    yp = np.zeros((n, MAX_CELLS, MAX_CELLS), dtype=np.float32)
    yk = np.zeros((n, 1), dtype=np.float32)
    mask = np.zeros((n, MAX_CELLS, MAX_CELLS), dtype=np.float32)

    failed = 0
    loaded = 0

    for fpath in files:
        try:
            sample = np.load(fpath)
            x[loaded, :, :, 0] = pad(sample["enrich"], MAX_CELLS)
            x[loaded, :, :, 1] = pad(sample["coord_xn"], MAX_CELLS)
            x[loaded, :, :, 2] = pad(sample["coord_yn"], MAX_CELLS)

            yp[loaded] = pad(sample["flux"], MAX_CELLS)
            mask[loaded] = pad(sample["mask"], MAX_CELLS)
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

    print(f"Dataset loaded : {loaded} samples")
    return x, yp, yk, mask


# FNO spectral convolution layer — multiply truncated Fourier modes by learned weights
class SpectralConv2d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        scale = 1.0 / (width * width)
        self.width = width
        self.modes = modes
        self.w1 = nn.Parameter(scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        bsz, channels, height, width = x.shape
        xf = fft.rfft2(x)
        out = torch.zeros(
            bsz,
            self.width,
            xf.size(-2),
            xf.size(-1),
            dtype=torch.cfloat,
            device=x.device,
        )

        # clip to available frequency bins in case grid is smaller than MODES
        mx = min(self.modes, xf.size(-2))
        my = min(self.modes, xf.size(-1))

        # apply learned weights to lower-left and upper-left frequency blocks
        out[:, :, :mx, :my] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, :mx, :my], self.w1[:, :, :mx, :my]
        )
        out[:, :, -mx:, :my] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, -mx:, :my], self.w2[:, :, :mx, :my]
        )
        return fft.irfft2(out, s=(height, width))


# 2D Fourier Neural Operator — predicts flux field (phi) and keff scalar from input channels
class FNO2d(nn.Module):
    def __init__(self, modes, width, depth, pad, n_in):
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

        # predict flux field and normalize to unit L2 norm
        phi = self.phi_head(x.permute(0, 2, 3, 1)).squeeze(-1)
        phi = phi / (torch.norm(phi.reshape(phi.size(0), -1), dim=1).view(-1, 1, 1) + 1e-12)

        # global pool features for keff prediction; softplus keeps output positive
        k_mean = x.mean(dim=(-2, -1))
        k_max = x.amax(dim=(-2, -1))
        k_std = x.std(dim=(-2, -1))
        k = F.softplus(self.k_head(torch.cat([k_mean, k_max, k_std], dim=1))) + 1e-6
        return phi, k


def save_plots(img, title, filename, cmap="viridis"):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.colorbar(format="%.3g")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_hist(history, out_dir):
    metrics = [
        ("train_flux_loss", "val_flux_loss", "Flux Loss", "flux_loss.png"),
        ("train_total_loss", "val_total_loss", "Total Loss", "total_loss.png"),
        ("train_flux_rel_err", "val_flux_rel_err", "Relative Flux Error", "flux_rel_err.png"),
        ("train_k_rel_err", "val_k_rel_err", "Relative keff Error", "k_rel_err.png"),
    ]

    epochs = history.get("epoch", [])
    if not epochs:
        return

    for train_key, val_key, title, fname in metrics:
        train_vals = history.get(train_key, [])
        val_vals = history.get(val_key, [])

        plt.figure(figsize=(7, 5))
        plt.plot(epochs, train_vals, label="train")

        # skip None entries inserted when no val set was available
        if val_vals and any(v is not None for v in val_vals):
            valid_x = [ep for ep, v in zip(epochs, val_vals) if v is not None]
            valid_y = [v for v in val_vals if v is not None]
            plt.plot(valid_x, valid_y, label="val")

        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="fno_trained.pt")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to save plots")
    parser.add_argument("--data", default=None, help="Path to samples folder; overrides checkpoint path")
    parser.add_argument("--sample", type=int, default=None, help="Dataset index to plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, device)

    required = [
        "model_state", "x_mean", "x_std",
        "fno_modes", "fno_width", "fno_depth", "fno_pad",
        "n_in", "history", "samples_dir",
    ]
    for key in required:
        if key not in ckpt:
            raise KeyError(f"Missing checkpoint key: {key}")

    samples_dir = args.data if args.data is not None else ckpt["samples_dir"]
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    x, yp, yk, masks = load(samples_dir)

    x_mean = ckpt["x_mean"]
    x_std = ckpt["x_std"]

    model = FNO2d(
        modes=ckpt["fno_modes"],
        width=ckpt["fno_width"],
        depth=ckpt["fno_depth"],
        pad=ckpt["fno_pad"],
        n_in=ckpt["n_in"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_tensor = torch.tensor((x - x_mean) / x_std, dtype=torch.float32, device=device)

    # default to first validation sample; fall back to first train sample
    if args.sample is None:
        val_idx = ckpt.get("val_idx", [])
        train_idx = ckpt.get("train_idx", [])
        if len(val_idx):
            plot_idx = int(val_idx[0])
        elif len(train_idx):
            plot_idx = int(train_idx[0])
        else:
            plot_idx = 0
    else:
        plot_idx = int(args.sample)

    if plot_idx < 0 or plot_idx >= len(x):
        raise IndexError(f"sample index {plot_idx} out of range for dataset of size {len(x)}")

    with torch.no_grad():
        pred_phi, pred_k = model(x_tensor[plot_idx:plot_idx + 1])

    enrich = x[plot_idx, :, :, 0]
    mask = masks[plot_idx]
    true_phi = yp[plot_idx]
    pred_phi = pred_phi[0].cpu().numpy()
    abs_err = np.abs(pred_phi - true_phi)
    rel_err = abs_err / (np.abs(true_phi) + 1e-8)

    enrich_c = crop(enrich, mask)
    true_phi_c = crop(true_phi, mask)
    pred_phi_c = crop(pred_phi, mask)
    abs_err_c = crop(abs_err, mask)
    rel_err_c = crop(rel_err, mask)

    save_plots(enrich_c, "Enrichment", os.path.join(out_dir, "enrichment.png"), cmap="plasma")
    save_plots(true_phi_c, "True Flux", os.path.join(out_dir, "true_flux.png"))
    save_plots(pred_phi_c, "Predicted Flux", os.path.join(out_dir, "pred_flux.png"))
    save_plots(abs_err_c, "Absolute Error", os.path.join(out_dir, "abs_error.png"))
    save_plots(rel_err_c, "Relative Error", os.path.join(out_dir, "rel_error.png"))

    fig, axs = plt.subplots(1, 4, figsize=(22, 5))

    im0 = axs[0].imshow(enrich_c, cmap="plasma")
    axs[0].set_title("Enrichment")
    plt.colorbar(im0, ax=axs[0], format="%.3g")

    im1 = axs[1].imshow(true_phi_c, cmap="viridis")
    axs[1].set_title("True Flux")
    plt.colorbar(im1, ax=axs[1], format="%.3g")

    im2 = axs[2].imshow(pred_phi_c, cmap="viridis")
    axs[2].set_title("FNO Flux")
    plt.colorbar(im2, ax=axs[2], format="%.3g")

    im3 = axs[3].imshow(abs_err_c, cmap="viridis")
    axs[3].set_title("Absolute Error")
    plt.colorbar(im3, ax=axs[3], format="%.3g")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "main_result.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plot_hist(ckpt["history"], out_dir)

    print(f"Plotted sample index : {plot_idx}")
    print(f"True keff            : {yk[plot_idx, 0]:.6f}")
    print(f"Pred keff            : {pred_k.item():.6f}")
    print(f"Saved plots to       : {out_dir}/")
