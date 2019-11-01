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



alphabet_size = 4

prediction1 = np.array([1, 2, 1, 1])
prediction2 = np.array([3, 4, 1, 3])
assert len(prediction1) == alphabet_size
assert len(prediction2) == alphabet_size

initial_model_state = 0

# returns (pseudo)frequencies for each symbol
def get_model_prediction(model_state):
    if model_state == 0:
        return prediction1
    else:
        return prediction2

# updates the state of the model according to an observed symbol
def get_model_next_state(model_state, symbol):
    return model_state + 1






def C(number, symbol, numerators):
    denominator = numerators.sum()
    numerators_accum = np.insert(np.cumsum(numerators), 0, 0)
    q, r = np.divmod(number, numerators[symbol])
    return denominator * q + r + numerators_accum[symbol]
    #return ((number // numerators[symbol]) << n) + (number % numerators[symbol]) + numerators_accum[symbol]

def D(number, numerators):
    denominator = numerators.sum()
    numerators_accum = np.insert(np.cumsum(numerators), 0, 0)
    q, r = np.divmod(number, denominator)
    symbol = np.where(numerators_accum <= r)[0][-1]
    return symbol, numerators[symbol] * q + r - numerators_accum[symbol]
    #mask = 2**n - 1
    #symbol = np.where(numerators_accum <= number & mask)[0][-1]
    #return numerators[symbol] * (number >> n) + (number & mask) - numerators_accum[symbol], symbol

def encode(string):
    number = 1

    # collect predictions made by the model for every string prefix
    model_state = initial_model_state
    predictions = []
    for symbol in string:
        predictions.append(get_model_prediction(model_state))
        model_state = get_model_next_state(model_state, symbol)

    for symbol, prediction in zip(string[::-1], predictions[::-1]):
        number = C(number, symbol, prediction)
    return number

def decode(number):
    string = []

    model_state = initial_model_state
    while True:
        symbol, number = D(number, get_model_prediction(model_state))
        model_state = get_model_next_state(model_state, symbol)

        string.append(symbol)
        if number == 1:
            break
    return string
    # any sequence of 0s at the end of the string will be chopped off during 
    #   decoding due to the way I set up the termination condition
    # alternatively one could use an end-of-string special symbol

for trial in count():
    string = list(np.random.choice(alphabet_size, size=6))
    string[-1] = np.random.randint(1, alphabet_size) # string cannot end in 0

    encoding = encode(string)
    decoding = decode(encoding)
    assert decoding == string

    print('success on trial {}'.format(trial))
