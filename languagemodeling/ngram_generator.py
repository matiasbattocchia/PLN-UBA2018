from collections import defaultdict
import numpy as np


class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self._n = model._n
        self._probs = defaultdict(dict)

        for (ngram, count) in model.n0_counter.items():
            self._probs[ ngram[:-1] ][ ngram[-1] ] = count / model.n1_counter[ ngram[:-1] ]

        self._probs = dict(self._probs)

    def generate_token(self, prev_tokens=tuple()):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        cond_probs = self._probs[prev_tokens]

        return np.random.choice(list(cond_probs.keys()), p=list(cond_probs.values()))

    def generate_sent(self):
        """Randomly generate a sentence."""
        sent = ['<s>'] * (self._n - 1)

        while True:
            prev_tokens = [] if self._n == 1 else sent[-(self._n-1):]
            token = self.generate_token(tuple(prev_tokens))

            if token == '</s>' or len(sent) > 100:
                return sent[self._n-1:]

            sent.append(token)
