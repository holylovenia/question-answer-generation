from collections import defaultdict

class BoyerMoore:
    def __init__(self):
        pass

    class LastOccurence:
        def __init__(self, pattern):
            self.occurrences = defaultdict(lambda: -1)
            for i, v in enumerate(pattern):
                self.occurrences[v] = len(pattern) - i - 1

        def __call__(self, letter):
            return self.occurrences[letter]

    @staticmethod
    def find(text, pattern):
        last = BoyerMoore.LastOccurence(pattern)
        m = len(pattern)
        n = len(text)
        i = m - 1  # text index
        j = m - 1  # pattern index
        while i < n:
            if text[i] == pattern[j]:
                if j == 0:
                    return i
                else:
                    i -= 1
                    j -= 1
            else:
                l = last(text[i])
                i = i + m - min(j, 1+l)
                j = m - 1 
        return -1
