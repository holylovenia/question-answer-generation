from collections import defaultdict

def compute_last_occ(pattern):
    l = len(pattern)
    last_occ = defaultdict(lambda: l)

    for i, v in enumerate(pattern):
        last_occ[v] = l - i - 1

    return last_occ

def find(text, pattern):
    last_occ = compute_last_occ(pattern)
    skip = 0

    while len(text) - skip >= len(pattern):
        i = len(pattern) - 1
        while text[skip + i] == pattern[i]:
            if i == 0:
                return skip
            i -= 1
        
        skip += last_occ[text[skip + len(pattern) - 1]]

    return -1