import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import OrderedDict
from model import ModifiedResNet
import pickle
import pandas as pd
import numpy as np


class CIFAR10TestDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        data: array-like of shape (N, 3, 32, 32) or (N, 32, 32, 3),
              depending on how you saved it
        transform: torchvision transforms to apply
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        # Convert to a float Tensor
        # (If data is already a float32 / torch Tensor, adjust accordingly)
        if self.transform is not None:
            # Transforms normally expect a PIL image or torch Tensor in [C,H,W]
            img = self.transform(img)

        return img


def main():
    # 2) Load the model & weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModifiedResNet(num_blocks=[4,4,3], base_channels=64, num_classes=10, use_se=True).to(device)
    checkpoint = torch.load('best_model.pth', map_location='cuda')

    # Create a new OrderedDict where we rename each key
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        # Remove the "_orig_mod." prefix
        if k.startswith("_orig_mod."):
            new_k = k.replace("_orig_mod.", "")
        else:
            new_k = k
        new_state_dict[new_k] = v

    # Now load the renamed state dict
    model.load_state_dict(new_state_dict)
    model.eval()

    # 3) Load test data
    with open('cifar_test_nolabel.pkl','rb') as f:
        test_data = pickle.load(f)[b'data']
    print("Test data shape:", test_data.shape)

    # 4) Define transform
    transform_test = transforms.Compose([
        transforms.ToTensor(),  # if test_data is still in (H,W,C) numpy
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    # 5) Create dataset and dataloader
    test_dataset = CIFAR10TestDataset(test_data, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # 6) Inference
    all_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(dim=1)
            all_preds.extend(predicted.cpu().numpy())

    # 7) Export to CSV
    df = pd.DataFrame({
        "ID": range(len(all_preds)),
        "Labels": all_preds
    })
    df.to_csv("submission.csv", index=False)
    print("Saved predictions to submission.csv")

if __name__ == '__main__':
    main()