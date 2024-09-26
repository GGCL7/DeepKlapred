import torch
import torch.utils.data as Data
import numpy as np

# 蛋白质序列索引映射字典
protein_residue2idx = {
    '[PAD]': 0,
    '[CLS]': 1,
    '[SEP]': 2,
    'A': 3,   # Alanine
    'C': 4,   # Cysteine
    'D': 5,   # Aspartic acid
    'E': 6,   # Glutamic acid
    'F': 7,   # Phenylalanine
    'G': 8,   # Glycine
    'H': 9,   # Histidine
    'I': 10,  # Isoleucine
    'K': 11,  # Lysine
    'L': 12,  # Leucine
    'M': 13,  # Methionine
    'N': 14,  # Asparagine
    'P': 15,  # Proline
    'Q': 16,  # Glutamine
    'R': 17,  # Arginine
    'S': 18,  # Serine
    'T': 19,  # Threonine
    'V': 20,  # Valine
    'W': 21,  # Tryptophan
    'Y': 22,  # Tyrosine
}


def transform_protein_to_index(sequences, residue2idx):
    token_index = []
    for seq in sequences:
        seq_id = [residue2idx.get(residue, 0) for residue in seq]  # 使用get方法防止未知字符
        token_index.append(seq_id)
    return token_index


def pad_sequence(token_list, max_len=51):
    data = []
    for i in range(len(token_list)):
        token_list[i] = [protein_residue2idx['[CLS]']] + token_list[i]  # 在前面添加CLS标记
        n_pad = max_len - len(token_list[i])
        token_list[i].extend([protein_residue2idx['[PAD]']] * n_pad)
        data.append(token_list[i])
    return data


def read_protein_sequences_from_fasta(file_path):
    sequences = []
    labels = []
    sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    sequence = ''
                if '1' in line:
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
    return sequences, labels


def load_data_from_txt(file_path):
    sequences, labels = read_protein_sequences_from_fasta(file_path)
    indexed_sequences = transform_protein_to_index(sequences, protein_residue2idx)
    padded_sequences = pad_sequence(indexed_sequences)
    return padded_sequences, labels


def load_features_from_txt(feature_file_path):
    features = np.loadtxt(feature_file_path)
    return features


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, features, labels):
        self.input_ids = input_ids
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )
