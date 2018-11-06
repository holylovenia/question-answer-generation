from .base import find_all, flatten_nested_arrays, get_ranked_indices, lower_sentences

import math
import numpy as np


class SteinbergerJezekLSA():
  def __init__(self, stopwords=None):
    if stopwords is None:
      self.stopwords = []
    else:
      self.stopwords = stopwords

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
    
    # print(terms)
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
    self.tfidf = self.tfidf_transform(term_freq, lowered_sentences)
    # print(self.tfidf)

    self.u, self.sigma, self.vt = np.linalg.svd(self.tfidf)
    
    self.important_sentences = get_ranked_indices(self.compute_ranks(), self.top)
    
    return self.important_sentences