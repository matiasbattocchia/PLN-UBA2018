# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import math


class LanguageModel(object):

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        return 0.0

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        return -math.inf

    def log_prob(self, sents):
        result = 0.0
        for i, sent in enumerate(sents):
            lp = self.sent_log_prob(sent)
            if lp == -math.inf:
                return lp
            result += lp
        return result

    def cross_entropy(self, sents):
        log_prob = self.log_prob(sents)
        n = sum(len(sent) + 1 for sent in sents)  # count '</s>' events
        e = - log_prob / n
        return e

    def perplexity(self, sents):
        return math.pow(2.0, self.cross_entropy(sents))


from collections import Counter
import numpy as np

class NGram(LanguageModel):

    def ngrams(self, n, sent):
        # marcadores de comienzo y fin de oración
        _sent_ = ['<s>'] * (self._n - 1) + sent + ['</s>']

        for i in range(len(_sent_) - self._n + 1):
            yield tuple(_sent_[i:i+n])

    def __init__(self, n, sents, lower=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self._n = n
        self.n0_counter = Counter()
        self.n1_counter = Counter()

        for sent in sents:
            if lower:
                sent = [word.lower() for word in sent]

            # contando n-gramas y (n-1)-gramas
            self.n0_counter.update(self.ngrams(n,   sent))
            self.n1_counter.update(self.ngrams(n-1, sent))

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        if len(tokens) == self._n:
            return self.n0_counter[tokens]
        elif len(tokens) == self._n - 1:
            return self.n1_counter[tokens]

    def cond_prob(self, token, prev_tokens=tuple()):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        return self.cond_prob_ngram( prev_tokens + (token,) )

    def cond_prob_ngram(self, ngram):
        """Conditional probability of a token.

        tokens  -- tuple of tokens; i.e. (n-2, n-1, n)
        returns -- p(n | n-1, n-2) = count(n-2, n-1, n) / count(n-2, n-1)
        """
        prob = np.divide( self.n0_counter[ngram], self.n1_counter[ngram[:-1]] )

        if np.isnan(prob):
            return 0.0
        else:
            return prob

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        return np.prod( list(map( self.cond_prob_ngram, self.ngrams(self._n, sent) )) )

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        return np.sum( np.log2( list(map( self.cond_prob_ngram, self.ngrams(self._n, sent) )) ) )


class AddOneNGram(NGram):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        # call superclass to compute counts
        super().__init__(n, sents)

        # compute vocabulary
        self._voc = set( [ngram[-1] for ngram in self.n0_counter.keys()] )

        self._V = len(self._voc)  # vocabulary size

    def V(self):
        """Size of the vocabulary.
        """
        return self._V

    def cond_prob_ngram(self, ngram):
        """Conditional probability of a token.

        tokens  -- tuple of tokens; i.e. (n-2, n-1, n)
        returns -- p(n | n-1, n-2) = count(n-2, n-1, n) / count(n-2, n-1)
        """
        return np.divide( self.n0_counter[ngram] + 1, self.n1_counter[ngram[:-1]] + self._V)


class InterpolatedNGram(NGram):

    @staticmethod
    def ngrams(n, sent):
        # marcadores de comienzo y fin de oración
        _sent_ = ['<s>'] * n + sent + ['</s>']

        for i in range(len(_sent_) - n + 1):
            yield tuple(_sent_[i:i+n])

    def __init__(self, n, sents, gamma=None, addone=True, lower=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self._n = n

        if gamma is not None:
            # everything is training data
            train_sents = sents
        else:
            # 90% training, 10% held-out
            m = int(0.9 * len(sents))
            train_sents = sents[:m]
            held_out_sents = sents[m:]

        self.counters = [Counter() for _ in range(n+1)]

        for sent in train_sents:
            if lower:
                sent = [word.lower() for word in sent]

            for i in range(n+1):
                self.counters[i].update(self.ngrams(i, sent))

        self.counters[0][tuple()] -= len(train_sents)

        # compute vocabulary size for add-one in the last step
        self._addone = addone
        if addone:
            self._voc = set( [ngram[-1] for ngram in self.counters[-1].keys()] )
            self._V = len(self._voc)  # vocabulary size

        # compute gamma if not given
        if gamma is not None:
            self._gamma = gamma
        else:
            print('Computing gamma...')
            # use grid search to choose gamma
            min_gamma, min_p = None, float('inf')

            # WORK HERE!! TRY DIFFERENT VALUES BY HAND:
            for gamma in [100 + i * 50 for i in range(10)]:
                self._gamma = gamma
                p = self.perplexity(held_out_sents)
                print('  {} -> {}'.format(gamma, p))

                if p < min_p:
                    min_gamma, min_p = gamma, p

            print('  Choose gamma = {}'.format(min_gamma))
            self._gamma = min_gamma

    def count(self, ngram):
        """Count for an k-gram for k <= n.

        tokens -- the k-gram tuple.
        """
        return self.counters[len(ngram)][ngram]

    def cond_prob_ngram(self, ngram):
        """Conditional probability of a token.

        tokens  -- tuple of tokens; i.e. (n-2, n-1, n)
        returns -- p(n | n-1, n-2) = count(n-2, n-1, n) / count(n-2, n-1)
        """
        prob = np.divide( self.count(ngram), self.count(ngram[:-1]) )
        print('COND=',ngram,'/',ngram[:-1],'=',self.count(ngram),
                '/',self.count(ngram[:-1]),'=', prob)

        return 0.0 if np.isnan(prob) else prob

    def gamma_term(self, ngram, gamma=0.0):
        prob = np.divide( self.count(ngram), self.count(ngram) + gamma)
        print('GAMMA=',ngram,'/',ngram,'+',gamma,'=',self.count(ngram),
                '/',self.count(ngram),'+',gamma,'=', prob)

        return 0.0 if np.isnan(prob) else prob

    def add_one_cond_prob_ngram(self, ngram):
        """Conditional probability of a token.

        tokens  -- tuple of tokens; i.e. (n-2, n-1, n)
        returns -- p(n | n-1, n-2) = count(n-2, n-1, n) / count(n-2, n-1)
        """
        return np.divide( self.count(ngram) + 1, self.count(ngram[:-1]) + self._V )

    def cond_prob(self, token, prev_tokens=tuple()):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self._n
        assert len(prev_tokens) == n - 1

        tokens  = prev_tokens + (token,)
        probs   = []
        lambdas = []

        for i in range(n-1):
            probs.append( self.cond_prob_ngram(tokens[i:]) )
            lambdas.append( (1 - np.sum(lambdas)) * self.gamma_term(tokens[i:-1], self._gamma) )

        if self._addone:
            probs.append( self.add_one_cond_prob_ngram(tokens[n-1:]) )
        else:
            probs.append( self.cond_prob_ngram(tokens[n-1:]) )

        lambdas.append(1 - np.sum(lambdas))

        print('p=',probs)
        print('l=',lambdas)
        print('cond=',np.dot(lambdas,probs))
        print('----')
        return np.dot(lambdas, probs)
