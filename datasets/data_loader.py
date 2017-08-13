"""
A generic module for dataset creation
"""
from spacy.en import English
from collections import Counter
import numpy as np
import os

nlp = English()

class CreateDataset():

    def generate_tokens(self, description):
        doc = nlp(unicode(description, 'utf-8'))
        return [x.orth_ for x in doc]

    def prepare_vocabulary(self, data):
        unique_id = Counter()
        for doc in data:
            sample_tokens = self.generate_tokens(doc)
            for token in sample_tokens:
                unique_id.update({token: 1})
        return unique_id

    def create_threshold(self, counter_dict):
        return Counter(el for el in counter_dict.elements() if counter_dict[el] > 5)

    def create_token2id_dict(self, token_list):
        return {v: k for k, v in enumerate(token_list)}

    def create_dataset(self, x, y, folder_path):
        data_tokens = []
        vocab_full = self.prepare_vocabulary(x)
        vocab_threshold = self.create_threshold(vocab_full)
        token2id = self.create_token2id_dict(list(vocab_threshold))
        token2id['_UNK'] = len(token2id) + 1
        id2token = {k: v for k, v in enumerate(token2id)}
        label2id = {v: k for k, v in enumerate(list(set(y)))}
        id2label = {k: v for k, v in enumerate(label2id)}
        labels = [label2id[item] for item in y]
        for doc in x:
            sample_tokens = self.generate_tokens(doc)
            data_tokens.append([token2id.get(y, token2id['_UNK']) for y in sample_tokens])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(os.path.join(folder_path, 'samples_encoded'), data_tokens)
        np.save(os.path.join(folder_path, 'token2id'), token2id)
        np.save(os.path.join(folder_path, 'id2token'), id2token)
        np.save(os.path.join(folder_path, 'label2id'), label2id)
        np.save(os.path.join(folder_path, 'id2label'), id2label)
        np.save(os.path.join(folder_path, 'labels_encoded'), labels)
        return data_tokens, labels

class LoadData():

    def __init__(self):
        return None

    def load_data_from_path(self, folder_path):
        try:
            samples_encoded = np.load(os.path.join(folder_path,'samples_encoded.npy'))
            labels_encoded = np.load(os.path.join(folder_path, 'labels_encoded.npy'))
            token2id = np.load(os.path.join(folder_path, 'token2id.npy'))
            label2id = np.load(os.path.join(folder_path, 'label2id.npy'))
            return samples_encoded, labels_encoded, token2id, label2id
        except:
            print 'No dataset exists in the specified path.'
            return None

