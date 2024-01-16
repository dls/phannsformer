import collections
import torch
from torch import nn
from torch.nn import functional as F
from phannsformer import Phannsformer
import unittest

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class TestData(torch.utils.data.Dataset):
    def __init__(self):
        pass
    def __getitem__(self, index):
        x = torch.randint(0, 25, (1,))
        y = 1 if x < 12 else 2
        return x.repeat(5), torch.tensor(y).repeat(5)
    def __len__(self):
        return 1000000

train_loader = torch.utils.data.DataLoader(dataset=TestData(), batch_size=20)
model = Phannsformer(64, 8, 2).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
min_learning_rate = 1e-4

mae_loss = nn.L1Loss()

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).int(), y.to(device).int()

        # Add noise
        nPos = torch.randint(0, 10, (X.shape[0], X.shape[1]), device=device).int()
        nV   = torch.randint(0, 24, (X.shape[0], X.shape[1]), device=device).int()
        nX = torch.clone(X)
        nX[nPos == 0] = nV[nPos == 0]
        # nX[:,4] = 1

        # Remove noise
        predX = model.denoise(nX).transpose(1, 2)
        noiseLoss = loss_fn(predX, X)
        noiseLoss.backward()

        # Compute prediction error
        pred = model.classify(X).transpose(1, 2)
        classLoss = loss_fn(pred, y)
        classLoss.backward()

        # Backpropagation
        optimizer.step()
        optimizer.zero_grad()

        if batch % 200 == 0:
            accuracyC = torch.sum(torch.argmax(pred, dim=1) == y).float() / (pred.shape[0] * pred.shape[2])
            accuracyN = torch.sum(torch.argmax(predX, dim=1) == X).float() / (X.shape[0] * X.shape[1])
            lossC, lossN, current = classLoss.item(), noiseLoss.item(), (batch + 1) * len(X)
            pct = (current/size) * 100
            print(f"accuracy: {accuracyC:>7f}; {accuracyN:>7f}; classify: {lossC:>7f}; noise: {lossN:>7f}  [{current:>5d}/{size:>5d}] {batch} {pct:>3f}")

        if batch != 0 and batch % 600 == 0:
            print("here")
            pass

        # if batch != 0 and batch % 600 == 0:
        #     print(torch.argmax(pred, dim=1))
        #     print()
        #     print(y)
        #     print("yo")
        #     print(torch.argmax(pred, dim=1) == y)
        #     print(torch.sum(torch.argmax(pred, dim=1) == y))
        #     print("yo")
        #     print()
        #     print(pred)
        #     exit()

        if batch != 0 and batch % 600 == 0:
            print(predX.shape)
            print(X.shape)
            print()
            print(torch.argmax(predX, dim=1))
            print()
            print(X)
            # print()
            # print(nPos == 0)
            print("yo")
            print(torch.argmax(predX, dim=1) == X)
            print(torch.sum(torch.argmax(predX, dim=1) == X))
            print("yo")
            print()
            print(predX)
            exit()


epochs = 2
learning_rate = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=min_learning_rate, T_max=epochs)
for t in range(epochs):
    train(train_loader, model, loss_fn, optimizer)
    learning_rate.step()
