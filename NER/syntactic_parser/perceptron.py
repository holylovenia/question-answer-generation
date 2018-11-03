from collections import defaultdict

import pickle

# https://gist.github.com/syllog1sm/10343947#file-gistfile1-py-L252

class Perceptron(object):
    def __init__(self, classes=None):
        self.classes = classes
        self.weights = {}
        self._totals = defaultdict(int)
        self._timestamps = defaultdict(int)
        self.i = 0

    def predict(self, features):
        scores = self.score(features)
        return max(self.classes, key=lambda clas: (scores[clas], clas))

    def score(self, features):
        all_weights = self.weights
        scores = dict((clas, 0) for clas in self.classes)

        for feature, value in features.items():
            if value == 0:
                continue
            if feature not in all_weights:
                continue

            weights = all_weights[feature]
            for clas, weight in weights.items():
                # Dot product
                scores[clas] += value * weight

        return scores

    def update(self, target, output, features):
        def update_feature(c, f, w, v):
            param = (f, c)
            self._totals[param] += (self.i - self._timestamps[param]) * w
            self._timestamps[param] = self.i
            self.weights[f][c] = w + v

        self.i += 1
        if output == target:
            return None

        for f in features:
            weights = self.weights.setdefault(f, {})
            update_feature(target, f, weights.get(target, 0.0), 1.0)
            update_feature(output, f, weights.get(output, 0.0), -1.0)

    def average_weight(self):
        for feature, weights in self.weights.items():
            new_weights = {}

            for clas, weight in weights.items():
                param = (feature, clas)
                total = self._totals[param]
                total += (self.i - self._timestamps[param]) * weight
                averaged = round(total / float(self.i), 3)
                if averaged:
                    new_weights[clas] = averaged
                
            self.weights[feature] = new_weights

    def save(self, path):
        pickle.dump(self.weights, open(path, 'wb'))

    def load(self, path):
        self.weights = pickle.load(open(path, 'rb'))
            


