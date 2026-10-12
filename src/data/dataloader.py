from torch.utils.data import DataLoader

def make_loader(dataset, batch_size=8, shuffle=False, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
