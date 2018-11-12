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

        if 'CLASSPATH' not in os.environ:
            os.environ['CLASSPATH'] = ".{}".format(sep)
        os.environ['CLASSPATH'] += "{}{}".format(stanford_parser_path, sep)
        
        self.parser = stanford.StanfordParser(model_path="{}/englishPCFG.ser.gz".format(stanford_parser_path))

    def parse(self, sentence):
        return self.parser.raw_parse((sentence))
