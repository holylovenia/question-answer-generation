import os

from nltk.parse import stanford

class StanfordParser():
    def __init__(self, stanford_parser_path):
        os.environ['CLASSPATH'] += stanford_parser_path + ";"
        self.parser = stanford.StanfordParser(model_path="{}/englishPCFG.ser.gz".format(stanford_parser_path))

    def parse(self, sentence):
        return self.parser.raw_parse((sentence))
