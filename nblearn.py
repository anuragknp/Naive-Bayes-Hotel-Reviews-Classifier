import sys
import os
import json
from nbclassifier import NBClassifier


def handle_folds(path, class_tuple, document):
    for name in os.listdir(path):
        if os.path.isdir(os.path.join(path, name)) and 'fold' in name:
            fold_path = os.path.join(path, name)
            for file_name in os.listdir(fold_path):
                if os.path.isfile(os.path.join(fold_path, file_name)):
                    document.append((os.path.join(fold_path, file_name), class_tuple))


def get_documents(base_path):
    document = []

    negative_path = os.path.join(base_path, 'negative_polarity')
    positive_path = os.path.join(base_path, 'positive_polarity')
    negative_deceptive_path = os.path.join(negative_path, 'deceptive_from_MTurk')
    negative_truthful_path = os.path.join(negative_path, 'truthful_from_Web')
    positive_deceptive_path = os.path.join(positive_path, 'deceptive_from_MTurk')
    positive_truthful_path = os.path.join(positive_path, 'truthful_from_TripAdvisor')

    handle_folds(negative_deceptive_path, ('negative', 'deceptive'), document)
    handle_folds(negative_truthful_path, ('negative', 'truthful'), document)
    handle_folds(positive_deceptive_path, ('positive', 'deceptive'), document)
    handle_folds(positive_truthful_path, ('positive', 'truthful'), document)

    return document

if __name__ == '__main__':
    documents = get_documents(sys.argv[1])
    nb_classifier = NBClassifier()
    nb_classifier.train(documents)
    with open('nbmodel.txt', 'w') as out:
        out.write(json.dumps(nb_classifier.get_model()))

