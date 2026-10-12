from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class MVTecDataset(Dataset):
    def __init__(self, root, category, split, transform=None):
        self.root = Path(root) / category
        self.split, self.transform = split, transform
        if not self.root.exists():
            raise FileNotFoundError(f"MVTec AD dataset not found at {self.root}. Put it under data/raw/mvtec/.")
        self.items = []
        if split == "train":
            paths = sorted((self.root / "train" / "good").glob("*.png"))
            self.items = [(p, None, 0) for p in paths]
        else:
            for defect in sorted((self.root / "test").iterdir()):
                for p in sorted(defect.glob("*.png")):
                    label = int(defect.name != "good")
                    mask = None if not label else self.root / "ground_truth" / defect.name / f"{p.stem}_mask.png"
                    self.items.append((p, mask, label))
        if not self.items:
            raise FileNotFoundError(f"No {split} images found under {self.root}.")

    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        path, mask_path, label = self.items[index]
        image = Image.open(path).convert("RGB")
        original_size = image.size[::-1]
        if self.transform: image = self.transform(image)
        mask = torch.zeros(original_size, dtype=torch.float32)
        if mask_path and mask_path.exists():
            mask = torch.from_numpy(__import__('numpy').array(Image.open(mask_path).convert('L')) > 0).float()
        return {"image": image, "label": torch.tensor(label), "mask": mask, "path": str(path)}
