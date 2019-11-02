import numpy as np
from time import time
from models import dummy_model_class, markov_model_class, ppm_model_class

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

def encode(symbols, model):
    model.reset()
    predictions = []
    for symbol in symbols:
        predictions.append(model.predict())
        model.observe(symbol)

    number = 1
    for symbol, prediction in reversed(tuple(zip(symbols, predictions))):
        number = C(number, symbol, prediction)
    return number

def decode(number, model):
    model.reset()
    symbols = ()
    while True:
        symbol, number = D(number, model.predict())
        model.observe(symbol)
        symbols += (symbol,)
        if symbol == 0: # null terminator
            break
    return symbols

if __name__ == '__main__':

    models = (dummy_model_class(), markov_model_class(0), markov_model_class(2), ppm_model_class(8))

    symbols = tuple(map(ord, open('iliad.txt').read(100000) + '\0'))
    for model in models:
        print('Training {} '.format(model), end='', flush=True)
        start = time()
        model.train(symbols)
        print('({:.2f} seconds)'.format(time() - start))
    print()

    symbols = tuple(map(ord, open('odyssey.txt').read(100) + '\0'))
    for model in models:
        print(model)
        start = time()
        encoding = encode(symbols, model)
        print('Encoded in {} bits ({:.2f} seconds)\n'.format(encoding.bit_length(), time() - start))
        assert symbols == decode(encoding, model)

    while True:
        symbols = tuple(map(ord, input('Input: ') + '\0'))
        start = time()
        encoding = encode(symbols, model)
        print('Encoded in {} bits ({:.2f} seconds)\n'.format(encoding.bit_length(), time() - start))
        assert symbols == decode(encoding, model)
