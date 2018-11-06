from nltk.cluster.util import cosine_distance
from operator import itemgetter
from .page_rank import PageRank

import numpy as np


class TextRank:
  def __init__(self, pagerank=None, top=2, stopwords=None):
    if pagerank is None:
      self.pagerank = PageRank()
    else:
      self.pagerank = pagerank
    self.top = top
    if stopwords is None:
      self.stopwords = []
    else:
      self.stopwords = stopwords

  def sentence_similarity(self, sentence1, sentence2):
    """
    sentence1 and sentence2 = ['word1', 'word2', ...]
    """
        
    sentence1 = [word.lower() for word in sentence1]
    sentence2 = [word.lower() for word in sentence2]
    all_words = list(set(sentence1 + sentence2))
    
    vector1 = [0] * len(all_words)
    vector2 = vector1.copy()
    
    # 1st sentence's vector building
    for word in sentence1:
      if word in self.stopwords:
        continue # do nothing
        
      # add 1 to vector's index corresponding to word
      vector1[all_words.index(word)] += 1
    
    # 2nd sentence's vector building
    for word in sentence2:
      if word in self.stopwords:
        continue # do nothing
        
      # add 1 to vector's index corresponding to word
      vector2[all_words.index(word)] += 1
            
    return 1 - cosine_distance(vector1, vector2)

  def sentence_similarity_matrix(self, sentences):
    """
    sentences = [['word1, 'word2', ...], [...], ...]
    """
    
    # create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
      for j in range(len(sentences)):
          if i == j:
              continue # do nothing
              
          similarity_matrix[i][j] = self.sentence_similarity(sentences[i], sentences[j])
            
    # row-wise matrix normalization to penalize longer sentences
    for i in range(len(similarity_matrix)):
      similarity_matrix[i] /= similarity_matrix[i].sum()

    return similarity_matrix

  def rank_text(self, sentences):
    """
    sentences = a list of sentences [[w11, w12, ...], [w21, w22, ...], ...]
    top = how may sentences the summary should contain
    stopwords = a list of stopwords
    """
    
    similarity_matrix = self.sentence_similarity_matrix(sentences)
    
    sentence_ranks = self.pagerank.page_rank(similarity_matrix)
    
    # sort sentence ranks
    ranked_sentence_indices = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    selected_sentences = ranked_sentence_indices[:self.top]
    
    return selected_sentences