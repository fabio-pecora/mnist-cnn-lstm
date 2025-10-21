import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataloaders(batch_size=64, data_dir="./data", num_workers=2, pin_memory=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_size = 50000
    val_size = len(full_train) - train_size
    train_data, val_data = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
