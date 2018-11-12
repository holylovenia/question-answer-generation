from nltk import sent_tokenize

from question_generation import GapSelector, Sentence
from sentence_selection import MultiWordPhraseExtractor, preprocess
from simple_nlg import SimpleNLG

class QuestionGenerator:
    def __init__(self, paragraph):
        self.paragraph = paragraph
        self.sentences = [sentence for sentence in sent_tokenize(paragraph)]

        self.ranker = MultiWordPhraseExtractor(window_size=5, top_keywords=7)
        self.rank = []

        self.questions = []

    def __rank_sentences(self, top=2):
        self.rank = self.ranker.summarize(preprocess(self.paragraph), top)

    def __generate_questions_from_sentence(self, sentence):
        s = Sentence(sentence)
        s.parse_constituents()
        s.extract_named_entities()

        gapper = GapSelector(s)
        candidates = gapper.process()

        for candidate in candidates:
            if not candidate["question"]:
                continue
            
            nlg = SimpleNLG(s, candidate)
            question = nlg.convert_to_question()

            if question:
                self.questions.append({
                    "sentence": candidate["sentence"],
                    "question_qw": question,
                    "question_gap": candidate["question"],
                    "answer": candidate["answer"]
                })

    def process(self, top_sentences=2):
        self.__rank_sentences(top_sentences)

        for r in self.rank:
            self.__generate_questions_from_sentence(self.sentences[r])
        
        return self.questions