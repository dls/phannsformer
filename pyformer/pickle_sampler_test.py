import collections
import torch
from torch import nn
from pickle_sampler import get_samplers
import unittest

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

train_loader, test_loader = get_samplers(1000, 10, device)

class TestStringMethods(unittest.TestCase):
    def test_frequencies(self):
        for batch, (X, y) in enumerate(train_loader):
            self.assertEqual(X.shape, torch.Size([1000, 10]))
            self.assertEqual(y.shape, torch.Size([1000, 10]))
            counter = collections.Counter(y[:, 1].numpy())
            print(counter)
            print(X[1, :])
            if batch == 5:
                break


if __name__ == '__main__':
    unittest.main()