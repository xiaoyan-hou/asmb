from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_anomaly_figure(image, mask, anomaly_map, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(image); axes[0].set_title("Image")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Ground truth")
    axes[2].imshow(anomaly_map, cmap="magma"); axes[2].set_title("Anomaly map")
    axes[3].imshow(image); axes[3].imshow(anomaly_map, cmap="magma", alpha=.45); axes[3].set_title("Overlay")
    for ax in axes: ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
