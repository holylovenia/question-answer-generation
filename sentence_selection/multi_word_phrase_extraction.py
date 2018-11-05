from nltk.tag.stanford import StanfordPOSTagger
from .stanford_postagger.stanford_wrapper import StanfordPOSTagger as StanfordPOSTaggerWrapper

import numpy as np
from .page_rank import PageRank


class MultiWordPhraseExtractor:
  def __init__(self, pagerank=PageRank(), window_size=3, top=5, stopwords=None):
    self.pagerank = pagerank
    self.window_size = window_size
    self.top = top
    if stopwords is None:
      self.stopwords = []
    else:
      self.stopwords = stopwords

  def add_postags(self, paragraph):
    postagger = StanfordPOSTaggerWrapper()
    
    postagged_paragraph = []
    for sentence in paragraph:
      postagged_sentence = []
      for word in sentence:
        if word in self.stopwords:
            continue # skip
        
        postagged_word = postagger.tag(word)
        postagged_sentence.append(postagged_word[0])
      postagged_paragraph.append(postagged_sentence)
    
    return postagged_paragraph

  def filter_words(self, postagged_words, tags=['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR']):
    postag_index = 1
    filtered_words = []
    
    for word in postagged_words:
      if word[postag_index] in tags:
        filtered_words.append(word)
    
    return filtered_words

  def flatten_nested_arrays(self, nested_arrays):
    return [item for sublist in nested_arrays for item in sublist]

  def find_all(self, element, array):
    indices = [i for i, x in enumerate(array) if x == element]
    return indices

  def do_they_occur_together(self, word1, word2, flattened_sentences):
    indices1 = find_all(word1, flattened_sentences)
    indices2 = find_all(word2, flattened_sentences)
    
    found = False
    for idx1, idx2 in zip(indices1, indices2):
      if abs(idx1 - idx2) <= self.window_size:
        found = True
        break
            
    return found

  def get_cooccurrence_matrix(self, all_words, flattened_sentences):
    cooccurrence_matrix = np.zeros((len(flattened_sentences), len(flattened_sentences)))
    
    for i in range(len(all_words)):
      for j in range(len(all_words)):
        if i == j:
          continue # do nothing
        
        if self.do_they_occur_together(all_words[i], all_words[j], flattened_sentences):
          cooccurrence_matrix[i][j] = 1
                
    return cooccurrence_matrix

  def pair_keywords(self, keywords, all_words, cooccurrence_matrix):
    word_index = 0   
    phrases = []
    
    for i in range(len(keywords)):
      for j in range(i + 1):
        if i == j:
          continue # do nothing
            
        idx1 = all_words.index(keywords[i])
        idx2 = all_words.index(keywords[j])
        
        if cooccurrence_matrix[idx1][idx2] == 1:
          phrases.append(all_words[min(idx1, idx2)][word_index] + ' ' + all_words[max(idx1, idx2)][word_index])
          break

    return self.phrases

  def multi_word_phrase_extraction(self, sentences):
    
    # annotate with POS tags
    postagged_sentences = self.add_postags(sentences)
    
    # flatten & filter words
    flattened_sentences = self.flatten_nested_arrays(postagged_sentences)
    all_words = list(set(flattened_sentences))
    filtered_words = self.filter_words(all_words)
    
    # get cooccurence matrix
    cooccurrence_matrix = self.get_cooccurrence_matrix(filtered_words, flattened_sentences)
    
    # compute pagerank
    keyword_ranks = pagerank.page_rank(cooccurrence_matrix)
    
    # sort keyword ranks
    ranked_keyword_indices = [item[0] for item in sorted(enumerate(keyword_ranks), key=lambda item: -item[1])]
    selected_keywords = ranked_keyword_indices[:top]
    
    self.keywords = itemgetter(*selected_keywords)(filtered_words)
    self.phrases = pair_keywords(self.keywords, filtered_words, cooccurrence_matrix)

    return self.phrases