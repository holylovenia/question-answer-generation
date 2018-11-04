# Relative import
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict

from .feature_extraction import extract_features
from .perceptron import Perceptron
from .util import DefaultList

from stanford_postagger import StanfordPOSTagger

SHIFT = 0
RIGHT = 1
LEFT = 2
MOVES = (SHIFT, RIGHT, LEFT)

class Parse(object):
    def __init__(self, n):
        self.n = n
        self.heads = [None] * (n - 1)
        self.labels = [None] * (n - 1)

        self.lefts = [DefaultList(0)] * n
        self.rights = [DefaultList(0)] * n

    def add(self, head, child, label=None):
        self.heads[child] = head
        self.labels[child] = label

        if child < head:
            self.lefts[head].append(child)
        else:
            self.rights[head].append(child)

class Parser(object):
    def __init__(self, load_path=None):
        self.model = Perceptron(MOVES)
        if load_path is not None:
            self.model.load(load_path)
        self.tagger = StanfordPOSTagger()
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))

    def save(self, path):
        self.model.save(path)

    def parse(self, words):
        n = len(words)
        i = 2
        stack = [1]
        parse = Parse(n)
        tags = [self.tagger.tag(word)[0][1] for word in words]

        while stack or (i + 1) < n:
            features = extract_features(words, tags, i, n, stack, parse)
            scores = self.model.score(features)
            valid_moves = get_valid_moves(i, n, len(stack))
            output = max(valid_moves, key=lambda move: scores[move])
            i = transition(output, i, stack, parse)

        return tags, parse.heads
    
    def train_one(self, itn, words, gold_tags, gold_heads):
        n = len(words)
        i = 2
        x = 0; y = 0
        stack = [1]
        parse = Parse(n)
        tags = [self.tagger.tag(word)[0][1] for word in words]

        while stack or (i + 1) < n:
            features = extract_features(words, tags, i, n, stack, parse)
            scores = self.model.score(features)
            valid_moves = get_valid_moves(i, n, len(stack))
            output = max(valid_moves, key=lambda move: scores[move])
            i = transition(output, i, stack, parse)

            gold_moves = get_gold_moves(i, n, stack, parse.heads, gold_heads)
            y += 1
            try:
                assert gold_moves
            except:
                x += 1
                continue
            target = max(gold_moves, key=lambda move: scores[move])

            self.model.update(target, output, features)
            self.confusion_matrix[target][output] += 1

        return (x, y, len([i for i in range(n - 1) if parse.heads[i] == gold_heads[i]]))

def transition(move, i, stack, parse):
    if move == SHIFT:
        stack.append(i)
        return i + 1
    elif move == RIGHT:
        parse.add(stack[-2], stack.pop())
        return i
    elif move == LEFT:
        parse.add(i, stack.pop())
        return i
    # raise GrammarError("Unknown move: %d" % move)
    assert move in MOVES

def get_valid_moves(i, n, stack_depth):
    moves = []
    if (i + 1) < n:
        moves.append(SHIFT)
    if stack_depth >= 2:
        moves.append(RIGHT)
    if stack_depth >= 1:
        moves.append(LEFT)
    return moves

def get_gold_moves(n0, n, stack, heads, gold):
    def deps_between(target, others, gold):
        for word in others:
            if gold[word] == target or gold[target] == word:
                return True
        return False

    valid = get_valid_moves(n0, n, len(stack))
    if not stack or (SHIFT in valid and gold[n0] == stack[-1]):
        return [SHIFT]
    if gold[stack[-1]] == n0:
        return [LEFT]

    costly = set([m for m in MOVES if m not in valid])

    # If the word behind s0 is its gold head, Left is incorrect
    if len(stack) >= 2 and gold[stack[-1]] == stack[-2]:
        costly.add(LEFT)
    
    # If there are any dependencies between n0 and the stack, pushing n0 will lose them
    if SHIFT not in costly and deps_between(n0, stack, gold):
        costly.add(SHIFT)

    # If there are any dependencies between s0 and the buffer, popping s0 will lose them
    if deps_between(stack[-1], range(n0 + 1, n - 1), gold):
        costly.add(LEFT)
        costly.add(RIGHT)

    return [m for m in MOVES if m not in costly]
