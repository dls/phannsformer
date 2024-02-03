import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).int(), y.to(device).int()
            pred = model.classify(X).transpose(1, 2)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    batch_size = 500
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    loss_fn = nn.CrossEntropyLoss()
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    test(test_dataloader, model, loss_fn)



if __name__ == "__main__":
    sys.exit(main())