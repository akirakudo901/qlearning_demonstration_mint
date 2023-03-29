import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        # Initializes a new DNN.
        super(Net, self).__init__()
        
        # - an input layer of size four
        self.fc1 = nn.Linear(4, 8)
        # - one hidden layer with 8 neurons, and final output of 2
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
print(net)

# params = list(net.parameters())
# print(len(params))
# print(params[0].size())  # conv1's .weight

# 
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))

# output = net(input)
# target = torch.randn(10)  # a dummy target, for example
# target = target.view(1, -1)  # make it the same shape as output
# criterion = nn.MSELoss()

# loss = criterion(output, target)
# print(loss)



# net.zero_grad()     # zeroes the gradient buffers of all parameters

# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)

# loss.backward()

# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)

# # create your optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.01)

# # in your training loop:
# optimizer.zero_grad()   # zero the gradient buffers
# output = net(input)
# loss = criterion(output, target)
# loss.backward()
# optimizer.step()    # Does the update