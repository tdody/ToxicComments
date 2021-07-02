import numpy as np
import re

from keras.preprocessing.text import Tokenizer

from sotoxic.config import dataset_config, model_config
from sotoxic.data_helper import data_loader

class DataTransformer(object):

    def __init__(self, max_num_words, max_sequence_length, char_level):
        self.data_loader = data_loader.DataLoader()
        self.clean_word_dict = self.data_loader.load_clean_words(dataset_config.CLEAN_WORDS_PATH)
        self.train_df = self.data_loader.load_dataset(dataset_config.TRAIN_PATH)
        self.test_df = self.data_loader.load_dataset(dataset_config.TEST_PATH)

        self.max_num_words = max_num_words
        self.max_sequence_length = max_sequence_length
        self.char_level = char_level
        self.tokenizer = None

    def prepare_data(self):
        """
        load, clean, and tokenize data
        
        Outputs:
        -------
            train_sequences: list of tokens of the training set
            training_labels: numpy array of labels
            test_sequences: list of tokens of the test set
        """
        # fill empty values
        list_sentences_train = self.train_df["comment_text"].fillna("no comment").values
        list_sentences_test = self.test_df["comment_text"].fillna("no comment").values
        list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

        # clean data
        self.comments = [self.clean_text(text) for text in list_sentences_train]
        self.test_comments = [self.clean_text(text) for text in list_sentences_test]

        # tokenize data
        self.build_tokenizer(self.comments + self.test_comments)
        train_sequences = self.tokenizer.texts_to_sequences(self.comments)
        training_labels = self.train_df[list_classes].values
        test_sequences = self.tokenizer.texts_to_sequences(self.test_comments)

        return train_sequences, training_labels, test_sequences

    def clean_text(self, text, clean_wiki_tokens=True, drop_image=True):
        """"""
        pass

    def build_embedding_matrix(self, embeddings_index):
        """returns embedding matrix"""
        nb_words = min(self.max_num_words, len(embeddings_index))
        embedding_matrix = np.zeros((nb_words, 300))
        word_index = self.tokenizer.word_index
        null_words = open('null-word.txt', 'w', encoding='utf-8')

        for word, i in word_index.items():

            if i >= self.max_num_words:
                null_words.write(word + ', ' + str(self.word_count_dict[word]) + '\n')
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                null_words.write(word + ', ' + str(self.word_count_dict[word]) + '\n')
        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        return embedding_matrix

    def build_tokenizer(self, comments):
        """fit tokenizer on comments"""
        self.tokenizer = Tokenizer(num_words=self.max_num_words, char_level=self.char_level)
        self.tokenizer.fit_on_texts(comments)