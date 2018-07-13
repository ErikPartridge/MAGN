
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter

writer = SummaryWriter()
# the amount per ingredient gets encoded into a single line 
# then each flavor 

INPUT_SIZE = 105

INGREDIENT_EMBEDDING_SIZE = 20
class TasteTester(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TasteTester, self).__init__()
        self.embed = nn.Embedding(input_size, INGREDIENT_EMBEDDING_SIZE)
        self.linear = nn.Linear(INGREDIENT_EMBEDDING_SIZE, hidden_size)
        self.do = nn.Dropout(p=0.05)
        self.l1 = nn.Linear(3000, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.linear(x)
        x = F.softmax(x)
        x = self.do(x)
        x = x.view(x.numel())
        x = self.l1(x)
        x = F.relu(x)
        return x
      
      
tt = TasteTester(INPUT_SIZE, 150)

criterion = nn.MSELoss()
optimizer = optim.Adamax(tt.parameters())


###

import json

MAX_INDEX = 0
INGREDIENT_DICTIONARY = {}

with open('combined.json') as f:
  rcp = json.load(f)

rcps = []
targets = []
for recipe in rcp:
  targets.append(recipe["rating"])
  temp = [[],[]]
  for key in recipe["ingredients"]:
    if key in INGREDIENT_DICTIONARY:
      temp[0].append(INGREDIENT_DICTIONARY[key])
      temp[1].append(recipe["ingredients"][key])
    else:
      INGREDIENT_DICTIONARY[key] = MAX_INDEX + 1
      MAX_INDEX = MAX_INDEX + 1
      temp[0].append(INGREDIENT_DICTIONARY[key])
      temp[1].append(recipe["ingredients"][key])
  temp[0] = np.pad(np.array(temp[0]), (0, 20 - len(temp[0])), mode="constant")
  temp[1] = np.pad(np.array(temp[1]), (0, 20 - len(temp[1])), mode="constant")
  rcps.append(np.array(temp))

rcps = np.array(rcps)
ys = torch.from_numpy((np.array(targets) - 3.5) / 1.4)
print(ys)
print(len(INGREDIENT_DICTIONARY))

###

for epoch in range(40):  # loop over the dataset multiple times
    running_loss = 0.0
    for i in range(105):
        # zero the parameter gradients
        optimizer.zero_grad()
        data = rcps[i]
        data = np.delete(data, 1, 0)[0]
        # forward + backward + optimize
        outputs = tt(Variable(torch.from_numpy(data).long()))
        loss = criterion(outputs, Variable(torch.from_numpy(np.array([ys[i]])).float()))
        loss.backward()
        optimizer.step()
        
print(tt.embed.weight)
print(INGREDIENT_DICTIONARY.items())
writer.add_embedding(tt.embed.weight, metadata=INGREDIENT_DICTIONARY.values())
writer.close()
