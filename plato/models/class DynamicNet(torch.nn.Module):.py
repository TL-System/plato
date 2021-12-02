import torch
import math
import random
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter


class Decomposed_Linear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            #sigma=None,
            #psi=None,
            l1_thres=None) -> None:
        """
        super().__init__(self, in_features, out_features, bias, device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.psi = Parameter(torch.randn(()))  #psi
        self.sigma = Parameter(torch.randn(()))  #sigma
        self.l1_thres = l1_thres
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Decomposed_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #self.weight = Parameter(
        #    torch.empty((out_features, in_features), **factory_kwargs))
        self.sigma = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs))
        self.psi = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        print("++++++")
        #self.reset_parameters()

    def forward(self, input):
        # print("size of sigma", self.sigma.size())
        # print("size of psi", self.psi.size())
        # print("size of bias", self.bias.size())
        self.weight = F.linear(torch.ones(self.sigma.size()), self.sigma,
                               self.psi)
        #  print("size of weights", self.weight.size())
        # print("size of inputs", input.size())
        return F.linear(input, self.weight, self.bias)


"""
class DynamicNet(torch.nn.Module):
    def __init__(self):
        
        In the constructor we instantiate five parameters and assign them as members.
       
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        
        For the forward pass of the model, we randomly choose either 4, 5
        and reuse the e parameter to compute the contribution of these orders.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same parameter many
        times when defining a computational graph.
        
        y = self.a + self.b * x + self.c * x**2 + self.d
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x**exp
        return y
"""

# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 10)
y = torch.sin(x)

# Construct our model by instantiating the class defined above
model = Decomposed_Linear(10, 10)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD([list(model.parameters())[0]],
                            lr=1e-8,
                            momentum=0.9)
for t in range(30000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 2000 == 1999:
        print(t, loss.item())
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
#print("list of params", list(model.parameters())[2])  # 0 is sigma ; 1 is psi