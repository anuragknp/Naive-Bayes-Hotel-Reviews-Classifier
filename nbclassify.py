__author__ = 'anurag'
import sys
import os
import json
from nbclassifier import NBClassifier


def get_documents(path, document, level = 0):
    if level < 3:
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)):
                get_documents(os.path.join(path, name), document, level + 1)
    else:
        for name in os.listdir(path):
            if os.path.isfile(os.path.join(path, name)):
                document.append(os.path.join(path, name))


if __name__ == '__main__':
    document = []
    get_documents(sys.argv[1], document)
    nb_classifier = NBClassifier()

    with open('nbmodel.txt', 'r') as input_file:
        nb_classifier.load_model(json.loads(input_file.read()))

    with open('nboutput.txt', 'w') as out:
        for doc_path in document:
            try:
                label_a = nb_classifier.predict(doc_path, ("truthful", "deceptive"))
                label_b = nb_classifier.predict(doc_path, ("positive", "negative"))
                print >>out, label_a[1], label_b[1], doc_path
            except Exception as e:
                pass


