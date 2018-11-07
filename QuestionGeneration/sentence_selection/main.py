from nltk import RegexpTokenizer

import re

from .lsa import SteinbergerJezekLSA
from .text_rank import TextRank
from .multi_word_phrase_extraction import MultiWordPhraseExtractor

def preprocess(paragraph):
    # replace all dictionary phonetic with ''
    paragraph = re.sub('\/.*\ˈ.*\/', '', paragraph)
    
    # replace all japanese characters with ''
    paragraph = re.sub('[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf\u3400-\u4dbf]+', '', paragraph)
    paragraph = re.sub(r'[^\x00-\x7f]',r'', paragraph) 
    
    # replace dots in the center of words
    words = paragraph.split(' ')
    for i in range(len(words)):
        if(words[i].find('.') != len(words[i]) - 1 and words[i].find('.') != -1):
            words[i] = words[i].replace('.', '')
        if(words[i].find(',') != len(words[i]) - 1 and words[i].find(',') != -1):
            words[i] = words[i].replace(',', '')

    paragraph = ' '.join(words)
    
    data_i_j = paragraph.split('.')
    
    paragraph = []
    for k in range(len(data_i_j)):
        tokenizer = RegexpTokenizer('[\w\/\&\-\:]+', flags=re.UNICODE)

        token_list = tokenizer.tokenize(data_i_j[k])
        token_list = [token.strip() for token in token_list if len(token.strip()) > 1 or token.lower() == 'a']
    
        if token_list != []:
            paragraph.append(token_list)
        
    return paragraph

def rank_sentences(paragraph):
    tr_jaccard = TextRank(similarity='jaccard')
    tr_cosine = TextRank(similarity='cosine')
    mwpe = MultiWordPhraseExtractor()
    lsa_tfidf = SteinbergerJezekLSA(matrix_technique='tfidf')
    lsa_binary = SteinbergerJezekLSA(matrix_technique='binary')

    num_sentences = len(paragraph)

    ranks = [
        lsa_tfidf.summarize(paragraph, top=num_sentences),
        lsa_binary.summarize(paragraph, top=num_sentences),
        mwpe.summarize(paragraph, top=num_sentences),
        tr_jaccard.summarize(paragraph, top=num_sentences),
        tr_cosine.summarize(paragraph, top=num_sentences)
    ]

    score = {}
    for rank in ranks:
        for i, v in enumerate(rank):
            curr_score = num_sentences - i
            if v in score:
                score[v] += curr_score
            else:
                score[v] = curr_score
                
    sorted_score = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    return [i for i, v in sorted_score]

def get_ranked_sentences(paragraph):
    return rank_sentences(preprocess(paragraph))
    