import os
import pickle
import torch
from Bio import SeqIO

print("Generating pickle files...")

categories = ["HTJ", "HTJ_connector", "baseplate", "baseplate_upper", "collar_tail_fiber",
              "major_capsid", "major_tail", "minor_tail", "portal", "shaft", "tail_spike",
              "other"]

fasta_mapping = {'M': 1,
                 'S': 2,
                 'I': 3,
                 'V': 4,
                 'A': 5,
                 'L': 6,
                 'R': 7,
                 'T': 8,
                 'P': 9,
                 'E': 10,
                 'Y': 11,
                 'K': 12,
                 'H': 13,
                 'D': 14,
                 'Q': 15,
                 'W': 16,
                 'F': 17,
                 'G': 18,
                 'N': 19,
                 'C': 20,
                 'X': 21,
                 'J': 22,
                 'B': 23,
                 'Z': 24}

def map_fasta_file(file):
    return [[fasta_mapping[e] for e in seq_record.seq] for seq_record in SeqIO.parse("michelle_classes_05_Jan_2022/" + file, "fasta")]

def load_fasta_files(category, ns):
    files = [category + "_" + str(n) + ".fasta" for n in ns]
    return [torch.tensor(item, dtype=torch.int8) for sublist in [map_fasta_file(file) for file in files] for item in sublist]

training_data = [(categories.index(category), item) for category in categories for item in load_fasta_files(category, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
testing_data = [(categories.index(category), item) for category in categories for item in load_fasta_files(category, [11])]

with open('training_data.pkl', 'wb') as file:
    training_ys = [item[0] for item in training_data]
    training_xs = [item[1] for item in training_data]
    pickle.dump((training_xs, training_ys), file)
with open('testing_data.pkl', 'wb') as file:
    testing_ys = [item[0] for item in testing_data]
    testing_xs = [item[1] for item in testing_data]
    pickle.dump((testing_xs, testing_ys), file)

print("Done generating pickle files.")