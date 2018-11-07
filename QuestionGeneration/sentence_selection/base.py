def find_all(element, array):
  indices = [i for i, x in enumerate(array) if x == element]
  return indices

def flatten_nested_arrays(nested_arrays, lower=False):
  if lower is False:
    return [item for sublist in nested_arrays for item in sublist]
  else:
    return [item.lower() for sublist in nested_arrays for item in sublist]

def get_ranked_indices(scores, top):
  ranked_indices = [item[0] for item in sorted(enumerate(scores), key=lambda item: -item[1])]
  return ranked_indices[:top]

def jaccard_similarity(sentence1, sentence2):
  intersection = len(list(set(sentence1).intersection(sentence2)))
  union = (len(sentence1) + len(sentence2)) - intersection
  return float(intersection / union)

def lower_sentences(sentences):
  lowered_sentences = sentences.copy()

  for i in range(len(sentences)):
    for j in range(len(sentences[i])):
      lowered_sentences[i][j] = sentences[i][j].lower()

  return lowered_sentences