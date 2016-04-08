__author__ = 'anurag'
import json
import math
from nbvocabulary import NBVocabulary
from nbdocument import NBDocument


class NBClassifier(object):

    def __init__(self):
        super(NBClassifier, self).__init__()
        self.smoothing_factor = 1
        self.__document_count = 0
        self.__vocabulary = NBVocabulary(self.smoothing_factor)
        self.__class_map = dict()

    def increment_class_count(self, class_name, token_count):
        if class_name not in self.__class_map.keys():
            self.__class_map[class_name] = dict({"document_count": 0, "vocab_count": 0})

        self.__class_map[class_name]["vocab_count"] += token_count
        self.__class_map[class_name]["document_count"] += 1

    def train(self, docs):
        for doc in docs:
            tokens = NBDocument(doc[0]).fetch_tokens()
            self.__document_count += 1

            for class_name in doc[1]:
                self.__vocabulary.add(tokens, class_name)
                self.increment_class_count(class_name, len(tokens))

    def get_model(self):
        # return training modal
        model = {"class_map": json.dumps(self.__class_map),
                 "vocabulary": self.__vocabulary.get_vocab_dumps(),
                 "document_count": self.__document_count}

        return model

    def load_model(self, model):
        self.__class_map = json.loads(model["class_map"])
        self.__vocabulary.load_vocab(model["vocabulary"])
        self.__document_count = model["document_count"]

    def __calculate_prior_probability(self, class_name):
        if class_name not in self.__class_map.keys():
            return 0

        return float(self.__class_map[class_name]["document_count"])/self.__document_count

    def get_class_count(self, class_name):
        if class_name not in self.__class_map.keys():
            return 1

        return self.__class_map[class_name]["vocab_count"]

    def predict(self, doc_path, class_list):
        tokens = NBDocument(doc_path).fetch_tokens()
        arg_max = []
        
        for class_name in class_list:
            prior_probability = self.__calculate_prior_probability(class_name)
            #ignoring unknow class
            if prior_probability == 0:
                continue

            probability = math.log(prior_probability)
            class_cnt = self.get_class_count(class_name)

            for token in tokens:
                tkn_cnt = self.__vocabulary.get_token_class_count(token, class_name)

                probability += math.log(float(tkn_cnt) / (class_cnt + self.__vocabulary.count))

            if len(arg_max) == 0:
                arg_max = [probability, class_name]

            if probability > arg_max[0]:
                arg_max[0] = probability
                arg_max[1] = class_name

        return arg_max

