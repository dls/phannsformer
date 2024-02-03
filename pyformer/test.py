import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 500

# Create data loaders.
# sampler = RandomSampler(training_data, replacement=True, num_samples=batch_size)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        width = 1024
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, width),
            nn.ReLU(),
            nn.LayerNorm(width),
            nn.Dropout(0.6),

            nn.Linear(width, width),
            nn.ReLU(),
            nn.LayerNorm(width),
            nn.Dropout(0.6),

            nn.Linear(width, width),
            nn.ReLU(),
            nn.LayerNorm(width),
            nn.Dropout(0.6),

            nn.Linear(width, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
min_learning_rate = 1e-4

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            print(f'\tResult|true:{y}|pred:{pred}')
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 100
learning_rate = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=min_learning_rate, T_max=epochs)
for t in range(epochs):
    train(train_dataloader, model, loss_fn, optimizer)
    learning_rate.step()
    if t % 10 == 0:
        test(test_dataloader, model, loss_fn)

test(test_dataloader, model, loss_fn)
print("Done!")
