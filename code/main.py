
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import math
from dataset import *
from model import *


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, sequence_features, labels = batch
        input_ids, sequence_features, labels = input_ids.to(device), sequence_features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, sequence_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

# 评估函数
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, sequence_features, labels = batch
            input_ids, sequence_features, labels = input_ids.to(device), sequence_features.to(device), labels.to(device)

            outputs = model(input_ids, sequence_features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy



train_sequences, train_labels = load_data_from_txt('/Users/ggcl7/Desktop/硕士蛋白质/protein lysine lactylation site prediction/train.txt')
test_sequences, test_labels = load_data_from_txt('/Users/ggcl7/Desktop/硕士蛋白质/protein lysine lactylation site prediction/test.txt')


sequence_feature_train = load_features_from_txt('/Users/ggcl7/Desktop/硕士蛋白质/protein lysine lactylation site prediction/train_feature.txt')
sequence_feature_test = load_features_from_txt('/Users/ggcl7/Desktop/硕士蛋白质/protein lysine lactylation site prediction/test_feature.txt')

dataset_train = MyDataSet(train_sequences, sequence_feature_train, train_labels)
dataset_test = MyDataSet(test_sequences, sequence_feature_test, test_labels)

train_loader = Data.DataLoader(dataset_train, batch_size=64, shuffle=True)
test_loader = Data.DataLoader(dataset_test, batch_size=64, shuffle=False)
vocab_size = len(protein_residue2idx)
d_model = 256
d_ff = 512
n_layers = 2
n_heads = 4
seq_feature_dim = sequence_feature_train.shape[1]
max_len = 51


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PTMWithCrossAttention(vocab_size, d_model, d_ff, n_layers, n_heads, seq_feature_dim, max_len).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)


n_epochs = 100
best_accuracy = 0.0
best_model_path = "best_model.pth"

for epoch in range(n_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # 保存准确度最高的模型
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f'新最佳模型保存于 Epoch {epoch+1}，准确度: {best_accuracy:.4f}')

print(f'训练完成。最高准确度: {best_accuracy:.4f}，模型已保存到 {best_model_path}')