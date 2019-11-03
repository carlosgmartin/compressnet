import numpy as np
import torch
from collections import deque, Counter, defaultdict
from pdb import set_trace

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

class rnn_model_class(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.gru = torch.nn.GRU(
            input_size=256,
            hidden_size=100,
            num_layers=2
        )

        self.out = torch.nn.Sequential(
            torch.nn.Linear(100, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 256)
        )

    def train(self, symbols):

        seq_len = 100
        batch_size = 10

        optimizer = torch.optim.Adam(self.parameters())

        for iteration in range(10000):

            indices = torch.randint(len(symbols) - seq_len - 1, size=(batch_size,))
            sequences = torch.tensor([symbols[index:index + seq_len + 1] for index in indices]).T

            inputs = torch.eye(256)[sequences[:-1]]
            targets = sequences[1:]

            outputs = self.out(self.gru(inputs)[0])

            loss = torch.nn.functional.cross_entropy(outputs.flatten(0, 1), targets.flatten(0, 1))
            print('\r\x1b[2K\x1b[2m' + 'iteration {} loss: {:.4f}'.format(iteration, loss.item()) + '\x1b[0m', end='', flush=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print()

    def reset(self):
        self.hidden_state = torch.zeros((self.gru.num_layers, 1, self.gru.hidden_size))

    def predict(self):
        return self.out(self.hidden_state)[-1, 0].softmax(-1).detach().numpy()

    def observe(self, symbol):
        self.hidden_state = self.gru(torch.eye(256)[symbol][None, None, :], self.hidden_state)[1]

    def predict_all_(self, symbols):
        # WARNING: may not end up with exactly the same floating point values as applying predict-observe over the symbol sequence
        prediction = self.predict()
        output, self.hidden_state = self.gru(torch.eye(256)[torch.tensor(symbols)][:, None, :], self.hidden_state)
        return (prediction,) + tuple(self.out(output)[:, 0, :].softmax(-1).detach().numpy())

    def __str__(self):
        return 'rnn model'
