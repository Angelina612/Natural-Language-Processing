# CS 563: Natural Language Processing

## Assignment-4: Neural Machine Translation

### Submitted By

Angelina Shibu \
2001CS06

### Problem Statement

The assignment aims to implement a Neural Machine Translation (NMT) system using Transformers. The model will be trained with German as the source language and English as the target language.

### Dataset

The Multi30K German-English parallel corpus is used for training and evaluation.

- [Dataset Link](https://www.dropbox.com/scl/fo/jkwmeu4dkyrwm4d7bfa69/AMA9ojeZmWnkhCrWjMptueo?rlkey=q965fljtlve0fdpuvl0ofuq6p&st=1o31nn92&dl=0)

### Data Format

Each data example consists of a German sentence as the source and its corresponding English translation as the target.

### Implementation Details

#### Preprocessing

- Added `<sos>` (Start of Sentence) and `<eos>` (End of Sentence) tokens to the start and end of each sentence, respectively.
- Length of sentence taken as average length of sentences in training data.
- Longer sentences are truncated and shorted sentences are appended with `<pad>` token.
- Built the vocabulary consisting of words occurring more than 5 times in the entire training set.

#### Model Architecture

- Transformer-based NMT model with 4 encoder-decoder stacks and 4 attention heads.
- Word embedding dimensionality: 128
- Position-wise feed-forward network dimensionality: 512

#### Training

- Trained the model until convergence using CrossEntropyLoss as the loss function.
- Perplexity of the validation set is used as the stopping criteria.
- Run `main.py` to train the model.

### Results

- The model did not converge in 10 epochs
- Perplexity of the test set after 10 epochs: 23.929

### References

- [transformer_nmt.ipynb](https://colab.research.google.com/drive/1Eu8TIjdjRUyj9Km-3KXcttBtjnPMrhjq?usp=sharing)
