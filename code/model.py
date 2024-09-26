import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=52):  # max_len 为41
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(EmbeddingLayer, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.bi_gru = nn.GRU(d_model, d_model // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)

    def forward(self, input_ids):
        x = self.src_emb(input_ids)
        embeddings = self.pos_emb(x.transpose(0, 1)).transpose(0, 1)
        gru_output, _ = self.bi_gru(embeddings)
        return gru_output



class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.3):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x



class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim=32, v_dim=32, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(self, x1, x2, mask=None):
        batch_size, in_dim1 = x1.size()
        batch_size2, in_dim2 = x2.size()


        assert batch_size == batch_size2, "Batch sizes of x1 and x2 must match."


        q1 = self.proj_q1(x1).view(batch_size, self.num_heads, self.k_dim).permute(0, 2, 1)
        k2 = self.proj_k2(x2).view(batch_size, self.num_heads, self.k_dim).permute(0, 2, 1)
        v2 = self.proj_v2(x2).view(batch_size, self.num_heads, self.v_dim).permute(0, 2, 1)

        attn = torch.matmul(q1.transpose(1, 2), k2) / self.k_dim ** 0.5


        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)


        attn = F.softmax(attn, dim=-1)


        output = torch.matmul(attn, v2.transpose(1, 2))


        output = output.transpose(1, 2).contiguous().view(batch_size, -1)
        output = self.proj_o(output)

        return output


class PTMWithCrossAttention(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_layers, n_heads, seq_feature_dim, max_len=41):
        super(PTMWithCrossAttention, self).__init__()

        self.emb = EmbeddingLayer(vocab_size, d_model)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.pool = nn.AdaptiveMaxPool1d(1)


        self.fc_seq_feature = nn.Sequential(
            nn.Linear(seq_feature_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.5)
        )


        self.cross_attention = CrossAttention(in_dim1=d_model, in_dim2=d_model)


        self.fc_combined = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.6),
            nn.ReLU()
        )

        self.fc_output = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, input_ids, sequence_features):

        emb_out = self.emb(input_ids)
        trans_out = self.transformer_blocks(emb_out)
        pooled_output = self.pool(trans_out.transpose(1, 2)).squeeze(-1)
        seq_feature_out = self.fc_seq_feature(sequence_features)
        cross_att_output = self.cross_attention(pooled_output, seq_feature_out)
        fc_combined_out = self.fc_combined(cross_att_output)
        logits = self.fc_output(fc_combined_out)

        return logits
