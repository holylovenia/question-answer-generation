import os

from pntl.tools import Annotator

class SemanticRoleLabeler:
    def __init__(self, senna_path, stanford_parser_path):
        self.senna_path = os.path.abspath(senna_path)
        self.stanford_parser_path = os.path.abspath(stanford_parser_path)

        self.annotator = Annotator(senna_dir=self.senna_path, stp_dir=self.stanford_parser_path)

    def get_srl(self, sentence):
        if isinstance(sentence, str):
            sentence = sentence.split()

        return self.annotator.get_annoations(sentence)['srl']

    