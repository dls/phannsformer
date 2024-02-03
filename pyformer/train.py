import torch
from torch import nn
from pickle_sampler import get_samplers
from phannsformer import Phannsformer

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = Phannsformer(64, 8, 8).to(device)
train_loader, test_loader = get_samplers(192, 128, device)
print(model)

noise_loss_fn = nn.CrossEntropyLoss()
class_loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
min_learning_rate = 1e-4

def accuracy(pred, y):
    correct = torch.sum(torch.argmax(pred, dim=1) == y).float()
    total = pred.shape[0] * pred.shape[2]
    return correct / total

def train(dataloader, model, noise_loss_fn, class_loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).long(), y.to(device).long()

        # Add noise
        nPos = torch.randint(0, 10, (X.shape[0], X.shape[1]), device=device).long()
        nV   = torch.randint(0, 24, (X.shape[0], X.shape[1]), device=device).long()
        nX   = torch.clone(X)
        #nX   = torch.where(nPos == 0, nV, nX)
        nX[nPos == 0] = nV[nPos == 0]

        # Remove noise
        predX = model.denoise(nX).transpose(1, 2)
        noiseLoss = noise_loss_fn(predX, X)
        noiseLoss.backward()

        # Compute prediction error
        pred = model.classify(X).transpose(1, 2)
        classLoss = class_loss_fn(pred, y)
        classLoss.backward()

        # Backpropagation
        optimizer.step()
        optimizer.zero_grad()

        if batch % 200 == 0:
            accuracyC = accuracy(pred, y)
            accuracyN = accuracy(predX, X)
            lossC, lossN, current = classLoss.item(), noiseLoss.item(), (batch + 1) * len(X)
            pct = (current/size) * 100
            print(f"accuracy: {accuracyC:>7f}; {accuracyN:>7f}; classify: {lossC:>7f}; noise: {lossN:>7f}  [{current:>5d}/{size:>5d}] {batch} {pct:>3f}")

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

epochs = 10
learning_rate = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=min_learning_rate, T_max=epochs)
for t in range(epochs):
    train(train_loader, model, noise_loss_fn, class_loss_fn, optimizer)
    learning_rate.step()
