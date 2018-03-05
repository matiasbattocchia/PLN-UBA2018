"""Train an n-gram model.

Usage:
  train.py [-m <model>] -n <n> -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
                  inter: N-grams with interpolation smoothing.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import RegexpTokenizer

CORPUS_DIR = '../corpus'

PATTERN = r'''(?x)   # verbose regexps
      \d+[.:]\d+     # horas y números con decimales
    | (?:\w+\.)+     # abreviaturas
    | \w+            # palabras alfanuméricas
    | [^\w\s]+       # signos de puntuación
'''

from ngram import NGram, AddOneNGram, InterpolatedNGram


models = {
    'ngram': NGram,
    'addone': AddOneNGram,
    'inter': InterpolatedNGram,
}


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    corpus = PlaintextCorpusReader(CORPUS_DIR, '.*.txt', word_tokenizer=RegexpTokenizer(PATTERN))

    # train the model
    n = int(opts['-n'])
    # model = NGram(n, corpus.sents())
    model = models[opts['-m']](n, corpus.sents())

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
