import torch
import numpy as np
import en_core_web_sm

from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, data, max_len=None):
        self.data = data
        self.max_len = max_len
        self.pre_process()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]['tokens']
        label = self.data[idx]['label']

        input = torch.tensor([])
        for w in tokens:
            if w in self.vocab:
                encoding = self.vocab.index(w)
            else:
                encoding = self.vocab.index("UNK")
            input = torch.concat((input, torch.tensor([encoding])))

        return input, torch.Tensor(label, requires_grad=True)
            
        
    def pre_process(self):
        cnt = Counter()
        self.vocab = set()

        if self.max_len is None:
            self.max_len = int(np.mean([len(data['text']) for data in self.data]))

        nlp = en_core_web_sm.load()

        for i, data in enumerate(self.data):
            if i % 100 == 0:
                print("-", end="")

            text = data['text']
            
            doc = nlp(text)
            tokens = [w.text for w in doc]
            
            if(len(tokens) > self.max_len):
                tokens = tokens[:self.max_len]
            elif(len(tokens) < self.max_len):
                for _ in range(len(tokens), self.max_len):
                    tokens.extend(["PAD"] * (self.max_len - len(tokens)))

            self.data[i]['tokens'] = tokens

            cnt.update(tokens)

        for w, freq in cnt.items():
            if freq >= 5:
                self.vocab.add(w)
        self.vocab.add("UNK")
        self.vocab = list(self.vocab)
        print()

def train(model, dataloader, criterion, epochs=10, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        with tqdm(total=len(dataloader), unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = model(inputs)
                _, outputs = torch.max(outputs, 1)
                if model.output_size == 1:
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, labels.float())
                else:
                    loss = criterion(outputs.float(), labels.float())
                running_loss += loss.item()
                # loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=running_loss / (i + 1))
                tepoch.update()
            losses.append(running_loss / len(dataloader))
    return losses


def test(model, dataloader):
    predicted = []
    true = []

    with tqdm(total=len(dataloader), unit="batch") as tepoch:
        tepoch.set_description("Testing")
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                outputs = model(inputs)
                _, outputs = torch.max(outputs, 1)

                predicted.extend(outputs)
                true.extend(labels)

                tepoch.update()

    return predicted, true

def one_hot_encoding(word, vocab):
    encoding = torch.zeros(len(vocab))
    index = vocab.index(word)
    encoding[index] = 1
    return encoding 
    
        