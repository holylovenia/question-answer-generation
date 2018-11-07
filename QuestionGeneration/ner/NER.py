import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from nltk.tag.stanford import StanfordPOSTagger
from stanford_postagger.stanford_wrapper import StanfordPOSTagger as StanfordPOSTaggerWrapper

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

import scipy

import pickle

class NER:

  def __init__(self, model_filename):
    if model_filename == None:
      print("Please provide the model filename")
    else:
      self.loaded_model = pickle.load(open(model_filename, 'rb'))
      print("Load Model Success")


  def add_other_label2dataset(self, dataset):
    """
    Input format: [['word1', 'word2', ...], ...]
    """
    other_label_dataset = []
    for sent in dataset:
        sent_list = []
        for token in sent:
            sent_list.append((token, 'O'))
        other_label_dataset.append(sent_list)
    return other_label_dataset

  def add_postag2dataset(self, dataset):
    postagger = StanfordPOSTaggerWrapper()
    dataset_with_postag = []
    for sent in dataset:
        postagged_sent = []
        for index, (token, tag) in enumerate(sent):
            postagged_token = postagger.tag(token)
            postagged_sent.append((token, postagged_token[0][1], tag))
        dataset_with_postag.append(postagged_sent)
        
    return dataset_with_postag


  def word2features(self, sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    # Ortographic Feature, Word, POSTag & N-Gram
    features = {
        'word': word,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:2]': word[:2],
        'word[:3]': word[:3],
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.isupper()': word.isupper(),
        'postag': postag,
        'postag[:2]': postag[:2]
    }
    
    # Position
    features.update({
        'pos_front': i,
        'pos_end': len(sent) - i
    })
    
    # Bag Of Words
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
        
    if i < len(sent) - 1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

  def sent2features(self, sent):
      return [self.word2features(sent, i) for i in range(len(sent))]

  def sent2postag(self, sent):
      return [postag for token, postag, label in sent]

  def sent2labels(self, sent):
      return [label for token, postag, label in sent]

  def sent2tokens(self, sent):
      return [token for token, postag, label in sent]


  def predict_class_text(self, text):
    """
    Input 1 sentence with space as the separator between words 
    e.g: John Doe is the most handsome person in the world
    """
    sentence = text.split()
    other_label_dataset = self.add_other_label2dataset([sentence])
    postagged_data = self.add_postag2dataset(other_label_dataset)
    features = [self.sent2features(sent) for sent in postagged_data]
    predicted = self.loaded_model.predict(features)
    return predicted

  def predict_class_text_list(self, sentence_list):
    """
    Input format: [['word1', 'word2', ...], ...]
    """
    other_label_dataset = self.add_other_label2dataset(sentence_list)
    postagged_data = self.add_postag2dataset(other_label_dataset)
    features = [self.sent2features(sent) for sent in postagged_data]
    predicted = self.loaded_model.predict(features)
    return predicted

  def predict_marginal_class_text(self, text):
    """
    Input 1 sentence with space as the separator between words 
    e.g: John Doe is the most handsome person in the world
    """
    sentence = text.split()
    other_label_dataset = self.add_other_label2dataset([sentence])
    postagged_data = self.add_postag2dataset(other_label_dataset)
    features = [self.sent2features(sent) for sent in postagged_data]
    predicted = self.loaded_model.predict_marginals(features)
    return predicted