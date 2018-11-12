from nltk.tag.stanford import StanfordPOSTagger
from operator import itemgetter
from .base import find_all, flatten_nested_arrays, get_ranked_indices
from .page_rank import PageRank
from stanford_postagger.stanford_wrapper import StanfordPOSTagger as StanfordPOSTaggerWrapper

import numpy as np


class MultiWordPhraseExtractor:
  def __init__(self, pagerank=None, window_size=3, top_keywords=5, stopwords=None):
    if pagerank is None:
      self.pagerank = PageRank()
    else:
      self.pagerank = pagerank
    self.window_size = window_size
    self.top_keywords = top_keywords
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
        if word.lower() in self.stopwords:
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
    cooccurrence_matrix = np.zeros((len(all_words), len(all_words)))
    
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

    return phrases

  def multi_word_phrase_extraction(self, sentences):
    
    # annotate with POS tags
    postagged_sentences = self.add_postags(sentences)
    
    # flatten & filter words
    flattened_sentences = flatten_nested_arrays(postagged_sentences)
    all_words = list(set(flattened_sentences))
    filtered_words = self.filter_words(all_words)
    
    # get cooccurence matrix
    cooccurrence_matrix = self.get_cooccurrence_matrix(filtered_words, flattened_sentences)
    
    # compute pagerank
    keyword_ranks = self.pagerank.page_rank(cooccurrence_matrix)
    
    # sort keyword ranks
    selected_keywords = get_ranked_indices(keyword_ranks, self.top_keywords)
    
    self.keywords = itemgetter(*selected_keywords)(filtered_words)
    self.phrases = self.pair_keywords(self.keywords, filtered_words, cooccurrence_matrix)

    return self.phrases

  def summarize(self, sentences, top=2):
    self.top = top
    self.multi_word_phrase_extraction(sentences)

    all_single_words = []
    for i in range(len(self.phrases)):
      splitted_phrase = self.phrases[i].split()
      all_single_words.append(splitted_phrase)
    
    all_single_words = flatten_nested_arrays(all_single_words)
    all_single_words = list(set(all_single_words))
    
    sentence_scores = np.zeros((len(sentences), 1))
    for i in range(len(sentences)):
      for j in range(len(all_single_words)):
        if all_single_words[j] in sentences[i]:
          sentence_scores[i] = sentence_scores[i] + 1

    self.important_sentences = get_ranked_indices(sentence_scores, self.top)
    
    return self.important_sentences