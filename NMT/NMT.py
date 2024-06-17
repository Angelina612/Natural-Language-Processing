import torch
import math

import numpy as np
import seaborn as sns

from transformer_nmt import Transformer_NMT
from torch import nn, optim
from tqdm import tqdm
from constants import *
from matplotlib import pyplot as plt


class NMT:
    def __init__(self, src_vocab, trg_vocab):
    # model parameters
        embedding_dim = 128
        src_vocab_size = len(src_vocab)
        trg_vocab_size = len(trg_vocab)
        n_heads = 4
        n_layers = 4
        src_pad_idx = src_vocab['<pad>']
        ff_dim = 512
        max_len = 100
        dropout = 0.1

        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)


        self.model = Transformer_NMT(
            embedding_dim = embedding_dim,
            src_vocab_size = src_vocab_size,
            trg_vocab_size = trg_vocab_size,
            n_heads = n_heads,
            n_layers = n_layers,
            src_pad_idx = src_pad_idx,
            ff_dim = ff_dim,
            max_len = max_len,
            dropout = dropout,
            device = self.device,
        ).to(self.device)

    def train(self, train_loader, dev_loader, trg_pad_idx, epochs=5):
        # optimizer and loss criterion
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
        best_val_ppl = float('inf')
        best_model_state = None 

        # train function
        train_losses = []
        eval_losses = []

        for i in range(epochs):
            # training
            self.model.train()
            for _, (batch_src, batch_trg) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training Progress"):
                in_src = batch_src.to(self.device)
                out_trg = batch_trg.to(self.device)

                output = self.model(in_src, out_trg[:-1, :]) # trg_len, batch_size, trg_vocab_size
                output = output.reshape(-1, output.shape[2]) # trg_len*batch_size, trg_vocab_size
                out_trg = out_trg[1:].reshape(-1)
                optimizer.zero_grad()

                train_loss = criterion(output, out_trg)
                train_losses.append(train_loss)

                train_loss.backward()
                optimizer.step()

            # eval
            self.model.eval()
            for _, (batch_src, batch_trg) in tqdm(enumerate(dev_loader), total = len(dev_loader), desc="Evaluation Progress"):
                in_src = batch_src.to(self.device)
                out_trg = batch_trg.to(self.device)

                output = self.model(in_src, out_trg[:-1, :]) # trg_len, batch_size, trg_vocab_size
                output = output.reshape(-1, output.shape[2]) # trg_len*batch_size, trg_vocab_size
                out_trg = out_trg[1:].reshape(-1)

                eval_loss = criterion(output, out_trg)
                eval_losses.append(eval_loss)

            val_loss = sum(eval_losses)/len(eval_losses)
            val_ppl = math.exp(val_loss)

            if val_ppl >= best_val_ppl:
                print("Validation perplexity increased. Stopping training.")
                break
            
            best_val_ppl = val_ppl
            best_model_state = self.model.state_dict()

            print(f'Epoch: {i+1}/{epochs}')
            print(f'Training Loss: {sum(train_losses)/len(train_losses):,.3f}\tEvaluation Loss: {sum(eval_losses)/len(eval_losses):,.3f}')
            print(f'Training PPL: {math.exp(sum(train_losses)/len(train_losses)):,.3f}\tEvaluation PPL: {math.exp(sum(eval_losses)/len(eval_losses)):,.3f}')

            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

def translate(model, test_loader, trg_vocab):
    predicted_translations = []
    reference_translations = []

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for batch_src, batch_trg in test_loader:
            # Move data to device
            batch_src = batch_src.to(device)
            batch_trg = batch_trg.to(device)

            # Forward pass to generate translations
            output = model(batch_src, batch_trg)

            # Convert predicted indices to words
            predicted_sentences = []
            for indices in output.argmax(dim=-1).tolist():
                # Map indices to tokens using trg_vocab
                tokens = [trg_vocab[idx] for idx in indices]
                # Remove START_TOKEN from translation if present
                if tokens[0] == START_TOKEN:
                    tokens = tokens[1:]
                # Remove END_TOKEN from translation if present
                if END_TOKEN in tokens:
                    tokens = tokens[:tokens.index(END_TOKEN)]
                predicted_sentences.append(tokens)

            # Convert reference indices to words
            reference_sentences = []
            for indices in batch_trg.tolist():
                # Map indices to tokens using trg_vocab
                tokens = [trg_vocab[idx] for idx in indices]
                # Remove START_TOKEN from translation if present
                if tokens[0] == START_TOKEN:
                    tokens = tokens[1:]
                # Remove END_TOKEN from translation if present
                if END_TOKEN in tokens:
                    tokens = tokens[:tokens.index(END_TOKEN)]
                reference_sentences.append(tokens)

            # Add predicted and reference translations to lists
            predicted_translations.extend(predicted_sentences)
            reference_translations.extend(reference_sentences)
        

    return predicted_translations, reference_translations

# # Tokenization
# def tokenize_data(text):
#     return [tok for tok in text.split()]

# def translate_sentence(sentence, model, max_len, src_vocab, trg_vocab):
#     device = ('cuda' if torch.cuda.is_available() else 'cpu')
#     model.eval()
        
#     if isinstance(sentence, str):
#         tokens = [tok.lower() for tok in sentence.split()]
#     else:
#         tokens = [tok.lower() for tok in sentence]

#     tokens = [src_vocab[START_TOKEN]] + [src_vocab.get(tok, src_vocab[UNK_TOKEN]) for tok in tokens] + [src_vocab[END_TOKEN]]
        
#     src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
#     src_mask = model.make_src_mask(src_tensor)
#     src_mask = src_mask.type(torch.bool)  # Convert mask to boolean datatype
    
#     with torch.no_grad():
#         enc_src = model.transformer.encoder(src_tensor, src_mask)

#     trg_indexes = [trg_vocab[START_TOKEN]]
#     attention = []

#     for i in range(max_len):
#         trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
#         trg_mask = model.transformer.generate_square_subsequent_mask(trg_tensor.size(1)).to(device)

#         with torch.no_grad():
#             output, attn_weights = model.transformer.decoder(trg_tensor, enc_src, trg_mask, src_mask)

#         attention.append(attn_weights)
#         pred_token = output.argmax(2)[:,-1].item()
#         trg_indexes.append(pred_token)

#         if pred_token == trg_vocab[END_TOKEN]:
#             break
    
#     trg_tokens = [trg_vocab.itos[i] for i in trg_indexes]
    
#     return trg_tokens[1:], torch.cat(attention, dim=1)


# def display_attention(sentence, translation, attention):
#     # Plotting attention matrix
#     fig = plt.figure(figsize=(10, 8))
#     sns.heatmap(attention, cmap='bone', xticklabels=sentence, yticklabels=translation, annot=True, fmt=".2f")
#     plt.title('Attention Visualization')
#     plt.xlabel('Source Sentence')
#     plt.ylabel('Translated Sentence')
#     plt.show()
