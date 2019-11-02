import numpy as np
from collections import deque, Counter, defaultdict

class dummy_model_class:

    def train(self, symbols):
        pass

    def reset(self):
        pass

    def predict(self):
        return np.ones(256) / 256

    def observe(self, symbol):
        pass

    def __str__(self):
        return 'dummy model'

class markov_model_class:

    def __init__(self, order):
        self.order = order
        self.memory = deque(maxlen=self.order)

    def train(self, symbols):
        self.table = Counter(zip(*(symbols[i:] for i in range(self.order + 1))))

    def reset(self):
        self.memory.clear()

    def predict(self):
        if len(self.memory) == self.order:
            counts = np.array([
                self.table.get(tuple(self.memory) + (symbol,), 0) + 1
                for symbol in range(256)
            ])
            return counts / counts.sum()
        else:
            return np.ones(256) / 256

    def observe(self, symbol):
        self.memory.append(symbol)

    def __str__(self):
        return 'markov model of order {}'.format(self.order)

class ppm_model_class:

    def __init__(self, order):
        self.order = order
        self.memory = deque(maxlen=self.order)

    def train(self, symbols):
        self.table = defaultdict(lambda: defaultdict(lambda: 0))
        for j in range(self.order + 1):
            for i in range(len(symbols) - j):
                self.table[symbols[i:i+j]][symbols[i+j]] += 1

    def reset(self):
        self.memory.clear()

    def predict(self):
        prefix = tuple(self.memory)
        while prefix not in self.table:
            prefix = prefix[1:]

        seen = len(self.table[prefix])
        unseen = 256 - seen
        total = seen + sum(self.table[prefix].values())
        return np.array([
            self.table[prefix][symbol]
            if symbol in self.table[prefix] else
            seen / unseen
            for symbol in range(256)
        ]) / total

    def observe(self, symbol):
        self.memory.append(symbol)

    def __str__(self):
        return 'ppm model of order {}'.format(self.order)
