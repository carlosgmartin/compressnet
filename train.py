import sys
import torch
from itertools import count
from glob import glob
from pdb import set_trace
from datetime import datetime

class LSTMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            num_layers=2,
            input_size=256, 
            hidden_size=128,
            dropout=.1
        )
        self.outnet = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=128,
                out_features=128
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=128,
                out_features=256
            ),
        )
    def forward(self, x):
        return self.outnet(self.lstm(torch.eye(256)[x])[0])

class FixedPrefixModel(torch.nn.Module):
    pass

if __name__ == '__main__':

    batch_size = 5

    model = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters())

    try:
        # load from specified file
        checkpoint = torch.load(sys.argv[1])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    except FileNotFoundError:
        pass

    corpus = open('iliad.txt').read()

    for iteration in count():

        if iteration % 5 == 0:
            torch.save({
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()
            }, sys.argv[1])

        indices = torch.randint(len(corpus) - 100, size=(batch_size,))
        passages = [corpus[index:index + 100] for index in indices]

        symbols = torch.tensor([list(map(ord, passage)) for passage in passages]).T

        targets = symbols[1:]
        inputs = symbols[:-1]
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs.flatten(0, 1), targets.flatten(0, 1))
        print('loss: {}'.format(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
