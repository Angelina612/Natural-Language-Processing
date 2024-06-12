# %% [markdown]
# Angelina Shibu \
# 2001CS06

# %%
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        data = file.read().split('\n\n')
    return data


# %%
lines = load_dataset("NER-Dataset-Train.txt")
print(lines[40])

# %%
tags = set()
data = []

for line in lines:
    words_tags = line.split('\n')
    words = []
    tokenized_words = []

    for word_tag in words_tags:
        if word_tag == '':
            continue
        words = word_tag.split('\t')
        tag = words[-1]
        word = words[0]
        tags.add(tag)
        tokenized_words.append((word, tag))

    data.append(tokenized_words)

# %%
tags = list(tags)
print(len(tags),tags)

# %%
display(data[40])

# %%
from models import BigramHMM, TrigramHMM

# %%
import numpy as np
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def k_fold_cross_validation(data, k, Model):
    kf = KFold(n_splits=k, shuffle=True)

    combined_actual_tags = []
    combined_predicted_tags = []

    for i, (train_index, test_index) in enumerate(kf.split(data)):
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]

        model = Model(train_data, tags)
        model.train()
        
        predicted_tags = []
        actual_tags = []

        for sentence in test_data:
            words = [word for word, _ in sentence]
            tagged_words = model.viterbi(words)
            predicted_tags.extend([tag for _, tag in tagged_words])
            actual_tags.extend([tag for _, tag in sentence])

        predicted_tags = np.array(predicted_tags).flatten()
        actual_tags = np.array(actual_tags).flatten()

        print(f"Fold {i+1} results:")
        print(classification_report(actual_tags, predicted_tags))
        combined_actual_tags.extend(actual_tags)
        combined_predicted_tags.extend(predicted_tags)

    return combined_actual_tags, combined_predicted_tags

# %%
actual_tags, predicted_tags = k_fold_cross_validation(data, 5, BigramHMM)


# %%
print("Overall results for bigram model:")
print(classification_report(actual_tags, predicted_tags))
print("Confusion Matrix:")
sns.heatmap(confusion_matrix(actual_tags, predicted_tags, labels=tags), cmap='Greens', xticklabels=tags, yticklabels=tags)

# %%
actual_tags, predicted_tags = k_fold_cross_validation(data, 5, TrigramHMM)

# %%
print("Overall results for trigram model:")
print(classification_report(actual_tags, predicted_tags))
print("Confusion Matrix:")
sns.heatmap(confusion_matrix(actual_tags, predicted_tags, labels=tags), cmap='Greens', xticklabels=tags, yticklabels=tags)


