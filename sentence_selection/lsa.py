from .base import find_all, flatten_nested_arrays, get_ranked_indices, lower_sentences

import math
import numpy as np


class SteinbergerJezekLSA():
  def __init__(self, stopwords=None, matrix_technique='tfidf'):
    if stopwords is None:
      self.stopwords = []
    else:
      self.stopwords = stopwords

    self.matrix_technique = matrix_technique

  def get_terms_from_sentences(self, sentences, stopwords=None):
    if stopwords is None:
      stopwords = []
    
    all_words = list(set(flatten_nested_arrays(sentences, lower=True)))
    for word in all_words:
      if word in stopwords:
        all_words.remove(word)
    
    terms = {}
    for i in range(len(sentences)):
      for word in sentences[i]:
        if word.lower() in all_words:
          if word.lower() not in terms:
            terms[word.lower()] = [i]
          else:
            if i not in terms[word.lower()]:
              terms[word.lower()].append(i)
    
    return terms

  def build_term_frequency_matrix(self, terms, sentences):
    # only use keys that are present in > 1 sentence
    keys = [k for k in terms.keys()]
    keys.sort()
    
    term_freq = np.zeros((len(keys), len(sentences)))
    for i in range(len(keys)):
      for sentence_num in terms[keys[i]]:
        match_results = len(find_all(keys[i], sentences[sentence_num]))
        term_freq[i][sentence_num] = match_results / len(sentences[sentence_num])
            
    return term_freq

  def tfidf_transform(self, term_freq_matrix, sentences):
    tfidf = term_freq_matrix.copy()
    total_sentences = tfidf.shape[1]
    
    for row in range(len(tfidf)):
      sentences_with_term = sum(tf > 0 for tf in tfidf[row])
      idf = math.log(abs(total_sentences / sentences_with_term))
      for col in range(len(tfidf[row])):
        tfidf[row][col] = tfidf[row][col] * idf
            
    return tfidf

  def binary_transform(self, term_freq_matrix):
    binary_matrix = term_freq_matrix.copy()
    total_sentences = binary_matrix.shape[1]
    
    for row in range(len(binary_matrix)):
      for col in range(len(binary_matrix[row])):
        if binary_matrix[row][col] > 0:
          binary_matrix[row][col] = 1
            
    return binary_matrix

  def compute_ranks(self):
    min_dimensions = 3
    dimensions = max(min_dimensions, len(self.sigma))
    powered_sigma = tuple(s**2 if i < dimensions else 0.0 for i, s in enumerate(self.sigma))
    
    ranks = []
    for col_vector in self.vt.T:
      rank = math.sqrt(sum(s*v**2 for s, v in zip(powered_sigma, col_vector)))
      ranks.append(rank)
    
    return ranks

  def summarize(self, sentences, top=2):
    self.top = top
    lowered_sentences = lower_sentences(sentences)

    terms = self.get_terms_from_sentences(lowered_sentences)
    term_freq = self.build_term_frequency_matrix(terms, lowered_sentences)
    if self.matrix_technique == 'tfidf':
      self.matrix = self.tfidf_transform(term_freq, lowered_sentences)
    else:
      self.matrix = self.binary_transform(term_freq)

    self.u, self.sigma, self.vt = np.linalg.svd(self.matrix)
    
    self.important_sentences = get_ranked_indices(self.compute_ranks(), self.top)
    
    return self.important_sentences