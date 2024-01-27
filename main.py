import pandas as pd
from fastai.tabular.all import *
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
import torch.optim as optim

pd.options.display.float_format = '{:2.f}'.format
set_seed(12)
df = pd.read_csv("train.csv")
GET_ROW = 0
EEG_PATH = 'train_eegs/'
SPEC_PATH = 'train_spectrograms/'
n_coeff = 50 * 20 * 200
t_dep_list = []
t_indep_list = []
train = pd.read_csv('train.csv')
for ROW in range(200):
    row = train.iloc[ROW]

    t_dep_list.append([x / max(row[9:15]) for x in row[9:15]])
    eeg = pd.read_parquet(f'{EEG_PATH}{row.eeg_id}.parquet')
    eeg_offset = int( row.eeg_label_offset_seconds )
    eeg = eeg.iloc[eeg_offset*200:(eeg_offset+50)*200]
    t_indep_list.append(eeg.values.flatten())



t_dep = tensor(t_dep_list) 
t_indep = tensor(t_indep_list, dtype=torch.float)
print(t_dep)
print(t_indep)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_coeff, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

net = Net()
print(net)


loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
