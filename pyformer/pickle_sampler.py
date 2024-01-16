import os
import pickle
import torch
from torch.utils.data.dataloader import DataLoader

with open('testing_data.pkl', 'rb') as file:
    testing_data, testing_labels = pickle.load(file)

with open('training_data.pkl', 'rb') as file:
    training_data, training_labels = pickle.load(file)

class FastaDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    def __len__(self):
        return len(self.data)

class ClassBalancedSampler(torch.utils.data.Sampler):
    def __init__(self, labels):
        classes = set(labels)
        self.length = len(labels)
        self.num_classes = len(classes)
        self.class_indexes = [[] for _ in range(self.num_classes)]
        for index, value in enumerate(labels):
            self.class_indexes[value].append(index)
    def __len__(self):
        return self.length
    def __iter__(self):
        for i in range(self.length):
            random_class = torch.randint(0, self.num_classes, (1,))
            random_index = torch.randint(0, len(self.class_indexes[random_class]), (1,))
            yield self.class_indexes[random_class][random_index]

def get_samplers(batch_size, context_size, device):
    # TODO(dls): padding mask
    def pad_or_trim_e(e):
        if len(e) == context_size:
            return e
        if len(e) < context_size:
            return torch.cat((e, torch.zeros(context_size - len(e), dtype=torch.int8)))
        else:
            offset = torch.randint(0, len(e) - context_size, (1,))
            return e[offset:offset + context_size]
    def pad_or_trim(arr):
        xs = [pad_or_trim_e(e[0]) for e in arr]
        ys = [torch.tensor(e[1], dtype=torch.int8).repeat(context_size) for e in arr]
        return torch.stack(xs), torch.stack(ys)

    train_loader = DataLoader(dataset=FastaDataset(training_data, training_labels),
                              batch_size=batch_size, pin_memory=True, collate_fn=pad_or_trim,
                              sampler=ClassBalancedSampler(training_labels))
    test_loader = DataLoader(dataset=FastaDataset(testing_data, testing_labels), batch_size=batch_size, collate_fn=pad_or_trim)
    return train_loader, test_loader
