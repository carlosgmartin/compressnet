'''
https://www.jessicayung.com/lstms-for-time-series-in-pytorch/
https://github.com/pytorch/examples/blob/master/time_sequence_prediction/train.py
'''

import sys
import torch
from itertools import count
from glob import glob
from pdb import set_trace
from datetime import datetime

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            num_layers=2,
            input_size=256, 
            hidden_size=512,
            dropout=.1
        )
        self.outnet = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=512,
                out_features=512
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=512,
                out_features=256
            ),
        )
    def forward(self, x):
        return self.outnet(self.lstm(torch.eye(256)[x])[0])

if __name__ == '__main__':

    batch_size = 5

    model = Model()
    if len(sys.argv) > 1:
        # load existing model from file
        model_path = sys.argv[1]
        model.load_state_dict(torch.load(model_path))
    else:
        # create a new model
        model_path = 'models/{}'.format(datetime.now())

    optimizer = torch.optim.Adam(model.parameters())

    corpus = '\n\n'.join(open(path).read() for path in glob('iliad/*'))

    for iteration in count():

        if iteration % 5 == 0:
            torch.save(model.state_dict(), model_path)

        indices = torch.randint(len(corpus) - 100, size=(batch_size,))
        passages = [corpus[index:index + 100] for index in indices]

        symbols = torch.tensor([list(map(ord, passage)) for passage in passages]).T

        targets = symbols[1:]
        inputs = symbols[:-1]
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs.flatten(0, 1), targets.flatten(0, 1))
        print('iteration {} loss: {}'.format(iteration, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
