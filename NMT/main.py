import torch

from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu
from NMT import *
from constants import *



class TranslationDataset(Dataset):
    def __init__(self, src_path, trg_path):

        with open(src_path, 'r', encoding='utf-8') as f:
            self.src = f.readlines()

        with open(trg_path, 'r', encoding='utf-8') as f:
            self.trg = f.readlines()

        assert len(self.src) == len(self.trg)

        self.build_vocab()

    def __len__(self):
        return len(self.src)
    
    def build_vocab(self):
        global VOCAB_SRC, VOCAB_DST
        VOCAB_SRC = {}
        VOCAB_DST = {}
        self.max_len = 0
        tot = 0

        for sentence in self.src:
            for word in sentence.split():
                if word not in VOCAB_SRC:
                    VOCAB_SRC[word] = 0
                VOCAB_SRC[word] += 1
            tot += len(sentence.split())
            # self.max_len = max(self.max_len, len(sentence.split()))

        self.max_len = tot // len(self.src)
        
        for sentence in self.trg:
            for word in sentence.split():
                if word not in VOCAB_DST:
                    VOCAB_DST[word] = 0
                VOCAB_DST[word] += 1

        # filter out words with frequency less than 2
        VOCAB_SRC = {k: v for k, v in VOCAB_SRC.items() if v >= MIN_FREQ}
        VOCAB_DST = {k: v for k, v in VOCAB_DST.items() if v >= MIN_FREQ}

        SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]
        VOCAB_SRC = {word: int(idx) for idx, word in enumerate(SPECIAL_TOKENS + list(VOCAB_SRC.keys()))}
        VOCAB_DST = {word: int(idx) for idx, word in enumerate(SPECIAL_TOKENS + list(VOCAB_DST.keys()))}
       

    def __getitem__(self, idx):
        src_sentence = [START_TOKEN] + self.src[idx].split() + [END_TOKEN]
        trg_sentence = [START_TOKEN] + self.trg[idx].split() + [END_TOKEN]

        src_sentence = src_sentence[:self.max_len]
        trg_sentence = trg_sentence[:self.max_len]

        src = [ VOCAB_SRC.get(word, VOCAB_SRC[UNK_TOKEN]) for word in src_sentence]
        trg= [ VOCAB_DST.get(word, VOCAB_SRC[UNK_TOKEN]) for word in trg_sentence]

        src += [VOCAB_SRC[PAD_TOKEN]] * (self.max_len - len(src_sentence))
        trg += [VOCAB_DST[PAD_TOKEN]] * (self.max_len - len(trg_sentence))
        
        # return torch.tensor(src), torch.tensor(trg)
        return torch.tensor(src).long(), torch.tensor(trg).long()
    
     
if __name__ == "__main__":
    train_dataset = TranslationDataset('Data/train.de', 'Data/train.en')
    train_data_loader = DataLoader(train_dataset, batch_size=32)

    dev_dataset = TranslationDataset('Data/val.de', 'Data/val.en')
    dev_data_loader = DataLoader(dev_dataset, batch_size=32)

    trg_pad_idx = VOCAB_DST[PAD_TOKEN]
    max_len = train_dataset.max_len

    model = NMT(VOCAB_SRC, VOCAB_DST)
    model.train(train_data_loader, dev_data_loader, trg_pad_idx, epochs=10)
    model.save_model('model.pt')
    # model.load_model('model.pt')

    test_dataset = TranslationDataset('Data/test_2016_flickr.de', 'Data/test_2016_flickr.en')
    test_data_loader = DataLoader(test_dataset, batch_size=32)

    # pred, ref = translate(model.model, test_data_loader, VOCAB_DST)
    # bleu = corpus_bleu(ref, pred)
    # print("BLEU Score: ", bleu)

    # sentence = "Dein Wort ist eine Laterne für meine Füße und ein Licht auf meinem Weg."
    # translated_sentence = translate_sentence(sentence, model.model, max_len, VOCAB_SRC, VOCAB_DST)
    # print("Translated Sentence:", " ".join(translated_sentence))