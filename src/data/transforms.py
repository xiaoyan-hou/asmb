from torchvision import transforms

def image_transform(resize: int = 256, image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
