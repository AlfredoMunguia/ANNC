import numpy as np 
import math, torch, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os
import random



model = torch.nn.Sequential(nn.Linear(10, 1, bias=False))

with torch.no_grad():
    # model[0].weight = nn.Parameter(torch.ones_like(model[0].weight))
    model[0].weight[0, 0] = 2.
    # model[0].weight.fill_(3.)



for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)
