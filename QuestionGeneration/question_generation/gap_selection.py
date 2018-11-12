import re

from .util import BoyerMoore

class GapSelector:
    candidate_labels = ['NP', 'ADJP']

    def __init__(self, sentence):
        self.sentence = sentence
        self.entities = None
        self.parents = None
        self.candidates = None

    def get_candidate_phrases(self):
        if self.sentence.constituent_tree is None:
            self.sentence.parse_constituents()

        self.entities = next(iter(map(
            lambda x: list(x.subtrees(filter=lambda x: x.label() in GapSelector.candidate_labels)),
            self.sentence.constituent_tree)))
        self.parents = []

    def filter_candidates_with_ner(self, max_len=5):
        filtered_entities = []
        for entity in self.entities:
            has_ner = False
            has_broken_ner = False
            leaves = [re.sub(r'[^\w\s]', '', leaf) for leaf in entity.leaves()]
            if len(leaves) > max_len:
                continue

            i = BoyerMoore.find(self.sentence.words, leaves)
            
            for leaf in leaves:
                if self.sentence.ner_classes[i][1] == 'O':
                    i += 1
                    continue
                
                is_subset = False
                for ne, _ in self.sentence.named_entities:
                    if BoyerMoore.find(leaves, ne) != -1:
                        is_subset = True
                        break
                        
                if is_subset:
                    has_ner = True
                    has_broken_ner = False
                else:
                    has_broken_ner = True
                break
                
            if has_broken_ner:
                continue
            
            # Accept only candidates with NER
            # if not has_ner:
            #     continue
            
            filtered_entities.append(entity)
        self.entities = filtered_entities

    def fetch_parent(self, max_depth=2):
        filtered_entities = []
        self.parents = []
        for entity in self.entities:
            # # Check for arguments
            parent = entity
            
            for _ in range(max_depth):
                parent = parent.parent()
                if parent.label() == 'S':
                    break
            if parent.label() == 'S':
                self.parents.append(parent)
                filtered_entities.append(entity)
                continue

            # Check for prepositional phrases
            parent = entity.parent()
            if parent.label() == 'PP':
                for _ in range(10):
                    parent = parent.parent()
                    if parent.label() == 'S':
                        break
                if parent.label() == 'S':
                    filtered_entities.append(entity)
                    self.parents.append(parent)
            
            
        self.entities = filtered_entities

    def create_gaps(self):
        self.candidates = []
        for entity, parent in zip(self.entities, self.parents):
            s = self.sentence.sentence
            leaves = parent.leaves()
            first_word, last_word = leaves[0], leaves[-1]

            i = ' '.join(self.sentence.constituent_tree[0].leaves()).find(' '.join(leaves))
            prefix = ' '.join(self.sentence.constituent_tree[0].leaves())[:i].split(' ')
            first_word_occurence = prefix.count(first_word)
            last_word_occurence = prefix.count(last_word)

            bi, li = (0, 0)
            for _ in range(first_word_occurence + 1):
                bi = s.find(first_word, bi) + len(first_word)
            for _ in range(last_word_occurence + 1):
                li = s.find(last_word, li) + len(last_word)
            bi -= len(first_word)

            s = s[bi:li]

            # candidate_gap = re.sub(r".*?([\w'–]+)(?:\)+)", r"\1 ", str(entity)).strip()
            # candidate_gap = re.sub(r" '", "'", candidate_gap)
            candidate_gap = ' '.join(entity.leaves()).replace(" '", "'")

            gap = re.compile("\b{}\b".format(candidate_gap), re.I)
            gapped_sentence = re.sub("\\b{}\\b".format(candidate_gap), '_____', s)
            
            if candidate_gap != s.strip():
                self.candidates.append({
                    'sentence': s,
                    'question': gapped_sentence,
                    'answer': candidate_gap,
                    'entity': entity,
                    'parent': parent
                })
                
        return self.candidates

    def process(self):
        self.get_candidate_phrases()
        self.filter_candidates_with_ner()
        self.fetch_parent()
        return self.create_gaps()
