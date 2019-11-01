'''

https://arxiv.org/abs/1311.2540
https://cs.stackexchange.com/questions/49243/is-there-a-generalization-of-huffman-coding-to-arithmetic-coding
https://en.wikipedia.org/wiki/Asymmetric_numeral_systems#Range_variants_(rANS)_and_streaming

https://arxiv.org/abs/1811.08162
https://theinformaticists.com/2019/03/22/lossless-compression-with-neural-networks/

'''

import numpy as np
import math
from pdb import set_trace
from itertools import count




initial_model_state = []

def get_model_prediction(model_state):
    return np.ones(256) / 256

def update_model_state(model_state, symbol):
    return model_state + [symbol]






def C(number, symbol, probabilities):
    # quantize probabilities
    numerators = np.ceil(probabilities * 100).astype(int)
    denominator = numerators.sum()

    numerators_accum = np.insert(np.cumsum(numerators), 0, 0)
    q, r = np.divmod(number, numerators[symbol])
    return denominator * q + r + numerators_accum[symbol]

def D(number, probabilities):
    # quantize probabilities
    numerators = np.ceil(probabilities * 100).astype(int)
    denominator = numerators.sum()

    numerators_accum = np.insert(np.cumsum(numerators), 0, 0)
    q, r = np.divmod(number, denominator)
    symbol = np.where(numerators_accum <= r)[0][-1]
    return symbol, numerators[symbol] * q + r - numerators_accum[symbol]

def encode(string):
    # collect predictions made by the model for every string prefix
    model_state = initial_model_state
    predictions = []
    for symbol in string:
        predictions.append(get_model_prediction(model_state))
        model_state = update_model_state(model_state, symbol)

    number = 1
    for symbol, prediction in zip(string[::-1], predictions[::-1]):
        number = C(number, symbol, prediction)
    return number

def decode(number):
    string = []
    model_state = initial_model_state
    while True:
        symbol, number = D(number, get_model_prediction(model_state))
        model_state = update_model_state(model_state, symbol)
        string.append(symbol)
        if symbol == 0: # null terminator
            break
    return string

while True:
    encoding = encode(list(map(ord, input('Input:    ')))) # input is automatically null-terminated
    print('Encoding: {}'.format(encoding))
    print('Decoding: {}\n'.format(''.join(map(chr, decode(encoding)))))
