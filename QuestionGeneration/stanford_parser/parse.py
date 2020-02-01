import os

from platform import system

from nltk.parse import stanford

classpath_separator = {
    "Linux": ":",
    "Windows": ";"
}

class StanfordParser():
    def __init__(self, stanford_parser_path):
        sep = classpath_separator[system()]
        self.parser = stanford.StanfordParser(path_to_models_jar="{}/englishPCFG.ser.gz".format(stanford_parser_path))

    def parse(self, sentence):
        return self.parser.raw_parse((sentence))
