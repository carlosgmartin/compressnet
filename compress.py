import numpy as np
from time import time
from models import dummy_model_class, markov_model_class, ppm_model_class, rnn_model_class

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
    return symbol, int(numerators[symbol]) * q + r - int(numerators_accum[symbol])

def encode(symbols, model):
    model.reset()
    try:
        predictions = model.predict_all(symbols)
    except AttributeError:
        predictions = ()
        for symbol in symbols:
            predictions += (model.predict(),)
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

    models = (rnn_model_class(), dummy_model_class(), markov_model_class(0), markov_model_class(2), ppm_model_class(8))

    symbols = tuple(map(ord, open('iliad.txt').read(100000) + '\0'))
    print('training...\n' + 'model'.ljust(30) + 'seconds')
    for model in models:
        start = time()
        model.train(symbols)
        print(str(model).ljust(30) + '{:.2f}'.format(time() - start))
    print()

    symbols = tuple(map(ord, open('odyssey.txt').read(1000) + '\0'))
    print('testing...\n' + 'model'.ljust(30) + 'seconds'.ljust(10) + 'bits')
    for model in models:
        start = time()
        encoding = encode(symbols, model)
        print(str(model).ljust(30) + '{:.2f}'.format(time() - start).ljust(10) + '{}'.format(encoding.bit_length()))
        assert symbols == decode(encoding, model)
    print()

    while True:
        symbols = tuple(map(ord, input('input: ') + '\0'))
        start = time()
        encoding = encode(symbols, model)
        print('encoded to {} bits in {:.2f} seconds\n'.format(encoding.bit_length(), time() - start))
        assert symbols == decode(encoding, model)
