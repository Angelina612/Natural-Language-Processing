# Transformer based NMT - 1 #
# Based on: https://pytorch.org/tutorials/beginner/transformer_tutorial.html #
# Based on: https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb #
# Based on : https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/seq2seq_transformer/seq2seq_transformer.py #
# Using nn.Transformer directly #
# Next let's break it down with nn.TransformerEncoder and nn.TransformerDecoder #

# imports
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F


class Transformer_NMT(nn.Module):
    def __init__(self, embedding_dim, src_vocab_size, trg_vocab_size, n_heads, n_layers, src_pad_idx, ff_dim, max_len, dropout, device):
        super(Transformer_NMT, self).__init__()
        self.src_tok_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.src_pos_embedding = nn.Embedding(max_len, embedding_dim)
        self.trg_tok_embedding = nn.Embedding(trg_vocab_size, embedding_dim)
        self.trg_pos_embedding = nn.Embedding(max_len, embedding_dim)
        self.device = device

        self.transformer = nn.Transformer(
            d_model = embedding_dim,
            nhead = n_heads,
            num_encoder_layers = n_layers,
            num_decoder_layers = n_layers,
            dim_feedforward = ff_dim,
            dropout = dropout,
            )

        # output of transformer model is: [target_seq_length, batch_size, hid_dim=embedding_dim]
        self.fc_out = nn.Linear(embedding_dim, trg_vocab_size)
        # we are transformering it to get: [target_seq_length, batch_size, output_dim=trg_vocb_size]

        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx # this is to tell the model which tokens in src should be ignored (as it is a pad token)

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx # creating a BoolTensor
        return src_mask.to(self.device)
        # so essentially we are telling model to ignore the src positions which have pad token

    def forward(self, src, trg):
        src_seq_len, N = src.shape
        trg_seq_len, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N).to(self.device)
        ) # here expand will be expanded to a larger size
        trg_positions = (
            torch.arange(0, trg_seq_len).unsqueeze(1).expand(trg_seq_len, N).to(self.device)
        )

        src_embedding = self.dropout(self.src_tok_embedding(src) + self.src_pos_embedding(src_positions))
        trg_embedding = self.dropout(self.trg_tok_embedding(trg) + self.trg_pos_embedding(trg_positions))

        src_pad_mask = self.make_src_mask(src)
        # print(trg_seq_len)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(self.device)
        # print(trg_mask.shape)

        output = self.transformer(
            src = src_embedding,
            tgt = trg_embedding,
            src_key_padding_mask = src_pad_mask,
            tgt_mask = trg_mask,
        )
        output = self.fc_out(output)

        return output
