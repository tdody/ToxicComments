import pandas as pd
import numpy as np

class DataLoader(object):

    def __init__(self):
        pass

    def load_dataset(self, dataset_path):
        """load csv into pandas dataframe"""
        return pd.read_csv(dataset_path)

    def load_clean_words(self, clean_words_path):
        """load correct spelling dictionary, keys is misspelled word, value is correct spelling"""

        clean_word_dict = {}
        with open(clean_words_path, 'r', encoding="utf-8") as cl:
            for line in cl:
                line = line.strip('\n')
                typo, correct = line.split(',')
                clean_word_dict[typo] = correct
        return clean_word_dict

    def load_embedding(self, embedding_path):
        """return a dict whose key is word, value is pretrained word embedding"""
        embeddings_index = {}
        with open(embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                try:
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                except:
                    print("Err on ", values[:2])
        print('Total %s word vectors.' % len(embeddings_index))
        return embeddings_index

            