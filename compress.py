'''

Asymmetric numeral systems: entropy coding combining speed of Huffman coding with compression rate of arithmetic coding

    https://arxiv.org/abs/1311.2540
    https://cs.stackexchange.com/questions/49243/is-there-a-generalization-of-huffman-coding-to-arithmetic-coding

DeepZip: Lossless Data Compression using Recurrent Neural Networks

    https://arxiv.org/abs/1811.08162
    https://theinformaticists.com/2019/03/22/lossless-compression-with-neural-networks/

Example texts (train on one, compress the other):

http://classics.mit.edu/Homer/iliad.html
http://classics.mit.edu/Homer/odyssey.html

'''

from train import LSTMModel
import torch
import sys
import numpy as np
from collections import Counter, defaultdict
from pdb import set_trace
from time import time

# training set
corpus = tuple(map(ord, open('iliad.txt').read()))



def bad_model(prefix):
    if len(prefix) > 0 and prefix[-1] == ord('e'):
        # say e is more likely to follow an e
        freqs = np.ones(256) + np.eye(256)[ord('e')]
        return freqs / freqs.sum()
    else:
        freqs = np.ones(256)
        return freqs / freqs.sum()



def ngrams(string, order):
    return Counter(zip(*(string[i:] for i in range(order))))

def ngram_model(prefix, table=ngrams(corpus, 2), order=2):
    if len(prefix) < order:
        return np.ones(256) / 256
    else:
        freqs = np.array([table.get(
            prefix[-order + 1:] + (symbol,), 0) + 1 
            for symbol in range(256)
        ])
        return freqs / freqs.sum()



def ppm(string, order):
    counts = defaultdict(lambda: defaultdict(lambda: 0))
    for j in range(order):
        for i in range(len(string) - j):
            counts[string[i:i+j]][string[i+j]] += 1
    return counts

def ppm_model(prefix, table=ppm(corpus, 8), order=8):
    prefix = prefix[-order:]
    while prefix not in table:
        prefix = prefix[1:]

    total = len(table[prefix].keys()) + sum(table[prefix].values())
    return np.array([
        table[prefix][symbol]
        if symbol in table[prefix] else 
        len(table[prefix].keys()) / (256 - len(table[prefix].keys()))
        for symbol in range(256)
    ]) / total



model = LSTMModel()
model.load_state_dict(torch.load(sys.argv[1])['model'])
model.eval()

def lstm_model(prefix, model=model):
    if len(prefix) == 0:
        return np.ones(256) / 256
    return model(torch.tensor(prefix)[:, None])[-1, 0].softmax(-1).detach().numpy()








def C(number, symbol, probabilities):
    # quantize probabilities
    numerators = np.ceil(probabilities * 100).astype(int)
    denominator = numerators.sum()
    numerators_accum = np.insert(np.cumsum(numerators), 0, 0)

    q, r = divmod(number, int(numerators[symbol]))
    return int(denominator) * q + r + int(numerators_accum[symbol])

def D(number, probabilities):
    # quantize probabilities
    numerators = np.ceil(probabilities * 100).astype(int)
    denominator = numerators.sum()
    numerators_accum = np.insert(np.cumsum(numerators), 0, 0)

    q, r = divmod(number, int(denominator))
    symbol = numerators_accum.searchsorted(r, side='right') - 1
    # symbol = np.where(numerators_accum <= r)[0][-1]
    return symbol, int(numerators[symbol]) * q + r - int(numerators_accum[symbol])

def encode(string, model):
    symbols = tuple(map(ord, string)) + (0,) # null terminated
    number = 1
    for i in reversed(range(len(symbols))):
        number = C(number, symbols[i], model(symbols[:i]))
    return number

def decode(number, model):
    symbols = ()
    while True:
        symbol, number = D(number, model(symbols))
        if symbol == 0: # null terminator
            break
        symbols += (symbol,)
    string = ''.join(map(chr, symbols))
    return string

if __name__ == '__main__':

    string = open('odyssey.txt').read(100)

    for model in [bad_model, ngram_model, ppm_model, lstm_model]:
        print('Encoding with {}: '.format(model.__name__), end='', flush=True)
        start = time()
        encoding = encode(string, model)
        print('{} bits ({:.2f} seconds)'.format(encoding.bit_length(), time() - start))
        assert string == decode(encoding, model)

    while True:
        string = input('\nInput:    ')
        encoding = encode(string, model)
        print('Encoding: {} bits'.format(encoding.bit_length()))
        assert string == decode(encoding, model)
