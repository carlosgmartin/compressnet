'''

Asymmetric numeral systems: entropy coding combining speed of Huffman coding with compression rate of arithmetic coding

    https://arxiv.org/abs/1311.2540
    https://cs.stackexchange.com/questions/49243/is-there-a-generalization-of-huffman-coding-to-arithmetic-coding

DeepZip: Lossless Data Compression using Recurrent Neural Networks

    https://arxiv.org/abs/1811.08162
    https://theinformaticists.com/2019/03/22/lossless-compression-with-neural-networks/

Example text: http://classics.mit.edu/Homer/iliad.1.i.html

'''

import numpy as np
from collections import Counter
from pdb import set_trace

def bad_model(prefix):
    if len(prefix) > 0 and prefix[-1] == ord('e'):
        # more likely to be followed by another e
        freqs = np.ones(256) + np.eye(256)[ord('e')]
        return freqs / freqs.sum()
    else:
        freqs = np.ones(256)
        return freqs / freqs.sum()

window_size = 2
corpus = tuple(map(ord, open('iliad book 1').read()))
windows = zip(*(corpus[i:] for i in range(window_size)))
counter = Counter(windows)
def good_model(prefix):
    if len(prefix) < window_size:
        return np.ones(256) / 256
    else:
        freqs = np.array([counter.get(prefix[-window_size + 1:] + (symbol,), 1) for symbol in range(256)])
        return freqs / freqs.sum()










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
    #symbol = np.where(numerators_accum <= r)[0][-1]
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

    string = open('iliad book 2').read(1000)
    
    model = bad_model
    encoding = encode(string, model)
    print('Encoding with bad model:  {} bits'.format(encoding.bit_length()))
    assert string == decode(encoding, model)

    model = good_model
    encoding = encode(string, model)
    print('Encoding with good model: {} bits'.format(encoding.bit_length()))
    assert string == decode(encoding, model)

    while True:
        string = input('\nInput:    ')
        encoding = encode(string, model)
        print('Encoding: {} bits'.format(encoding.bit_length()))
        assert string == decode(encoding, model)
