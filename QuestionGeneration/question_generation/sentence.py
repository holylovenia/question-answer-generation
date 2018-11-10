from nltk import ParentedTree, RegexpTokenizer

from ner.NER import NER
from semantic_parser import SemanticRoleLabeler
from stanford_parser import StanfordParser

ner_model_path = 'models/ner_model.pkl'
senna_path = 'lib/senna'
stanford_parser_path = 'lib'

class Sentence:
    ner = NER(ner_model_path)
    parser = StanfordParser(stanford_parser_path)
    annotator = SemanticRoleLabeler(senna_path, stanford_parser_path)

    def __init__(self, sentence):
        self.sentence = sentence
        self.words = RegexpTokenizer(r'\w+').tokenize(sentence)
        
        self.ner_classes = None
        self.named_entities = None
        self.named_entities_dict = None
        self.constituent_tree = None
        self.srl = None

    def extract_named_entities(self):
        ner_classes = Sentence.ner.predict_class_text(' '.join(self.words))
        self.ner_classes = [(word, ner_class) for word, ner_class in zip(self.words, ner_classes[0])]

        named_entities = []
        named_entity = []
        prev_ner_class = ['', '']

        for (word, ner_class) in self.ner_classes:
            if ner_class == 'O':
                if named_entity:
                    named_entities.append((named_entity, prev_ner_class[1]))
                    named_entity = []
                continue
                
            n = ner_class.split('-')
            if n[0] == 'B':
                named_entity.append(word)
            elif n[0] == 'I' and prev_ner_class[1] == n[1]:
                named_entity.append(word)

            prev_ner_class = n
            
        if named_entity:
            named_entities.append((named_entity, prev_ner_class[1]))

        self.named_entities = named_entities
        
        self.named_entities_dict = {}
        for ne, label in named_entities:
            if label in self.named_entities_dict:
                self.named_entities_dict[label].add(' '.join(ne))
            else:
                self.named_entities_dict[label] = {' '.join(ne)}
            for n in ne:
                self.named_entities_dict[label].add(n)


    def parse_constituents(self):
        self.constituent_tree = list(ParentedTree.convert(list(Sentence.parser.parse(self.sentence))[0]))
        return self.constituent_tree

    def parse_srl(self):
        self.srl = Sentence.annotator.get_srl(self.sentence)
        return self.srl

