__author__ = 'anurag'
import json


class NBVocabulary(object):
    def __init__(self, smoothing_factor):
        super(NBVocabulary, self).__init__()
        self.__vocab = dict()
        self.__smoothing_factor = smoothing_factor

    @property
    def count(self):
        return len(self.__vocab)

    def get_vocab_dumps(self):
        return json.dumps(self.__vocab)

    def load_vocab(self, vocab):
        self.__vocab = json.loads(vocab)

    def add(self, tokens, class_name):
        for token in tokens:
            if token not in self.__vocab:
                self.__vocab[token] = dict()

            if class_name not in self.__vocab[token]:
                self.__vocab[token][class_name] = 0

            self.__vocab[token][class_name] += 1

    def print_vocabulary(self):
        print(self.__vocab)

    def get_token_class_count(self, token, class_name):
        if token not in self.__vocab:
            return 0 + self.__smoothing_factor

        if class_name not in self.__vocab[token]:
            return 0 + self.__smoothing_factor

        return self.__vocab[token][class_name] + self.__smoothing_factor