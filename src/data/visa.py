import csv
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class VisADataset(Dataset):
    def __init__(self, root, category, split, transform=None, split_csv=None):
        self.root=Path(root); self.category=category; self.split=split; self.transform=transform
        csv_path=Path(split_csv) if split_csv else self.root/'split_csv'/'1cls.csv'
        if not csv_path.exists(): raise FileNotFoundError(f'VisA split file not found: {csv_path}')
        self.items=[]
        with csv_path.open(newline='',encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if row['object'] != category or row['split'] != split: continue
                image=self.root/row['image']; mask=self.root/row['mask'] if row.get('mask') else None
                label=int(row['label'].lower() != 'normal')
                self.items.append((image,mask,label))
        if not self.items: raise FileNotFoundError(f'No VisA {split} images for {category} in {csv_path}')
    def __len__(self): return len(self.items)
    def __getitem__(self,index):
        path,mask_path,label=self.items[index]; image=Image.open(path).convert('RGB'); original_size=image.size[::-1]
        if self.transform: image=self.transform(image)
        mask=torch.zeros(original_size,dtype=torch.float32)
        if mask_path and mask_path.exists(): mask=torch.from_numpy((np.array(Image.open(mask_path).convert('L'))>0).astype('float32'))
        return {'image':image,'label':torch.tensor(label),'mask':mask,'path':str(path)}
