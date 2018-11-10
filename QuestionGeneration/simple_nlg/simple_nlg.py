from nltk.stem import WordNetLemmatizer
from py4j.java_gateway import JavaGateway, java_import

blank = '_____'

simple_nlg_path = "lib/SimpleNLG-4.4.8.jar"

simple_nlg_imports = [
    "simplenlg.framework.*",
    "simplenlg.lexicon.*",
    "simplenlg.realiser.english.*",
    "simplenlg.phrasespec.*",
    "simplenlg.features.*",
]

def init_simple_nlg():
    gateway = JavaGateway.launch_gateway(classpath=simple_nlg_path)
    for i in simple_nlg_imports:
        java_import(gateway.jvm, i)
    return gateway

class SimpleNLG:
    gateway = init_simple_nlg()
    lexicon = gateway.jvm.Lexicon.getDefaultLexicon()
    nlg_factory = gateway.jvm.NLGFactory(lexicon)
    realiser = gateway.jvm.Realiser(lexicon)
    wnl = WordNetLemmatizer()

    def __init__(self, sentence, candidate):
        self.named_entities_dict = sentence.named_entities_dict
        self.entity = candidate['entity']
        self.parent = candidate['parent']
        self.answer = candidate['answer']
        self.question = candidate['question']

    def convert_to_question(self):
        if self.entity.parent().label() == 'S':
            return self.convert_subject()
        elif self.entity.parent().parent().label() == 'S':
            return self.convert_object()
        elif self.entity.parent().label() == 'PP':
            return 

    def convert_subject(self):
        if self.answer in self.named_entities_dict['PER']:
            question = self.question.replace(blank, "who")
        else:
            question = self.question.replace(blank, "what")
        
        q = list(question)
        q[0] = q[0].upper()
        if q[-1] == '.':
            q[-1] = '?'
        else:
            q.append('?')
        return ''.join(q)

    def convert_object(self):
        p = SimpleNLG.nlg_factory.createClause()
        
        for child in self.parent:
            # Subject
            if child.label() == 'NP':
                p.setSubject(' '.join(child.leaves()))
            
            # Verb
            elif child.label() == 'VP':
                for gchild in child:
                    if gchild.label() in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                        p.setVerb(self.lemmatize_verb(gchild))
                        if gchild.label() in ['VB', 'VBG', 'VBP', 'VBZ']:
                            p.setFeature(self.gateway.jvm.Feature.TENSE, self.gateway.jvm.Tense.PRESENT)
                        elif gchild.label() in ['VBD', 'VBN']:
                            p.setFeature(self.gateway.jvm.Feature.TENSE, self.gateway.jvm.Tense.PAST)
                        break
                        
            elif child.label() == 'PP':
                p.addComplement(' '.join(child.leaves()))
                        
        if self.answer in self.named_entities_dict['PER']:
            p.setFeature(self.gateway.jvm.Feature.INTERROGATIVE_TYPE, self.gateway.jvm.InterrogativeType.WHO_OBJECT)
        else:
            p.setFeature(self.gateway.jvm.Feature.INTERROGATIVE_TYPE, self.gateway.jvm.InterrogativeType.WHAT_OBJECT)
            
        return self.realiser.realiseSentence(p)

    def convert_pp(self):
        p = SimpleNLG.nlg_factory.createClause()
        for child in self.parent:
            # Subject
            if child.label() == 'NP':
                p.setSubject(' '.join(child.leaves()))
            
            # Verb
            elif child.label() == 'VP':
                for gchild in child:
                    if gchild.label() in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                        p.setVerb(self.lemmatize_verb(gchild))
                        if gchild.label() in ['VB', 'VBG', 'VBP', 'VBZ']:
                            p.setFeature(self.gateway.jvm.Feature.TENSE, self.gateway.jvm.Tense.PRESENT)
                        elif gchild.label() in ['VBD', 'VBN']:
                            p.setFeature(self.gateway.jvm.Feature.TENSE, self.gateway.jvm.Tense.PAST)
                        break
                        
            elif child.label() == 'PP':
                p.addComplement(' '.join(child.leaves()))
                        
        if self.answer in self.named_entities_dict['PER']:
            p.setFeature(self.gateway.jvm.Feature.INTERROGATIVE_TYPE, self.gateway.jvm.InterrogativeType.WHO_OBJECT)
        else:
            p.setFeature(self.gateway.jvm.Feature.INTERROGATIVE_TYPE, self.gateway.jvm.InterrogativeType.WHAT_OBJECT)
            
        return self.realiser.realiseSentence(p)

    def parse_pp(self, subtree):
        

    @classmethod
    def lemmatize_verb(cls, verb):
        return SimpleNLG.wnl.lemmatize(verb.leaves()[0], pos='v')
