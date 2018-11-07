import os

from nltk.parse import stanford

class StanfordParser():
    def __init__(self):
        os.environ['CLASSPATH'] = 'stanford_models'
        self.parser = stanford.StanfordParser(model_path="stanford_models/englishPCFG.ser.gz")

    def parse(self, sentence):
        return self.parser.raw_parse((sentence))
