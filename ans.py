'''

Asymmetric numeral systems: entropy coding combining speed of Huffman coding with compression rate of arithmetic coding

    https://arxiv.org/abs/1311.2540
    https://cs.stackexchange.com/questions/49243/is-there-a-generalization-of-huffman-coding-to-arithmetic-coding

DeepZip: Lossless Data Compression using Recurrent Neural Networks

    https://arxiv.org/abs/1811.08162
    https://theinformaticists.com/2019/03/22/lossless-compression-with-neural-networks/

'''

import numpy as np



# should cache previous calls for efficiency? (e.g. repeated prefixes)
def model(prefix):
    if len(prefix) > 0 and prefix[-1] == ord('e'):
        freqs = np.ones(256)
        return freqs / freqs.sum()
    else:
        freqs = np.ones(256) + np.eye(256)[3]
        return freqs / freqs.sum()






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
    number = 1
    for i in reversed(range(len(string))):
        number = C(number, string[i], model(string[:i]))
    return number

def decode(number):
    string = []
    while True:
        symbol, number = D(number, model(string))
        if symbol == 0: # null terminator
            break
        string.append(symbol)
    return string

while True:
    encoding = encode(list(map(ord, input('Input:    '))) + [0]) # null-terminated
    print('Encoding: {}'.format(encoding))
    print('Decoding: {}\n'.format(''.join(map(chr, decode(encoding)))))
