import numpy as np 
import math, torch, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os
import random


# Network structure
class RitzNet(torch.nn.Module):
    def __init__(self, params):
        super(RitzNet, self).__init__()
        self.params = params
        self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])

    def forward(self, x):
        
        # print(self.linearIn)
        # print(self.linearIn(x))
        
        x = torch.tanh(self.linearIn(x)) # Match dimension
        # print(x)
        
        for layer in self.linear:
            # print(layer)
            x_temp = torch.tanh(layer(x))
            # print(layer(x))
            # print(x_temp)
            x = x_temp

        return torch.tanh(self.linearOut(x))
        #return self.linearOut(x)
                

def initWeightAndBias(model):
    with torch.no_grad():        
        model.linearIn.weight[0][0] = .5 
        model.linearIn.weight[1][0] = .5
        model.linearIn.bias[0] = .5 
        model.linearIn.bias[1] = .5
        
        model.linear[0].weight[0][0] = .5
        model.linear[0].weight[1][0] = .5
        model.linear[0].weight[0][1] = .5
        model.linear[0].weight[1][1] = .5
        model.linear[0].bias[0] = .5
        model.linear[0].bias[1] = .5
        
        
        model.linear[1].weight[0][0] = .5
        model.linear[1].weight[1][0] = .5
        model.linear[1].weight[0][1] = .5
        model.linear[1].weight[1][1] = .5
        model.linear[1].bias[0] = .5
        model.linear[1].bias[1] = .5
        
        
        model.linearOut.weight[0][0] = .5 
        model.linearOut.weight[0][1] = .5
        model.linearOut.bias[0] = .5 


    
    
def trainData(params):
    nInt = params["bodyBatch"]
    a    = params["aa"]
    b    = params["bb"]
    
    array = np.zeros([nInt,1])        
    
    for i in range(nInt):
        # array[i] =  np.random.rand(1)*(b-a) + a
        array[i] =  -1; #(i/nInt) *(b-a) + a
        
    return array


def testData(params):
    nQuad = params["numQuad"]
    a    = params["aa"]
    b    = params["bb"]
    
    array = np.zeros([nQuad,1])        

    data =  np.linspace(params["aa"],params["bb"],params["numQuad"])
    
    for i in range(nQuad):
        array[i] =  data[i]

    return array





def train(model,device,params,optimizer,scheduler):

    data1 = trainData(params)
    data1 = torch.from_numpy(data1).float().to(device)
    target = exact(data1).to(device)
    

    model.train()
    
    # output1 = model(data1)

    # print('target',target)
    # print('output1', output1)

    # print(output1)
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data.shape)
    #         print (name, param.data)
            
    # print("---------")


    for step in range(params["trainStep"]):
        output1 = model(data1)
        model.zero_grad()

        # print(model.parameters())
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print (name, param.data.shape)
        #         print (name, param.data)
        # print("---------")


        #print(model.linearOut.weight)
        
        #Loss function 1
        target = exact(data1).to(device)
        
        loss1 = 0.5*torch.mean( (output1-target) * (output1-target) )
        loss  = loss1              
        

        if step%params["writeStep"] == params["writeStep"]-1:
            with torch.no_grad():
                target = exact(data1)
                error = errorFun(output1,target,params)
                # print("Loss at Step %s is %s."%(step+params["preStep"]+1,loss.item()))
                print("Error at Step %s is %s."%(step+1,error))
            file = open("lossData.txt","a")
            file.write(str(step+1)+" "+str(error)+"\n")


            
        if 10*(step+1)%params["trainStep"] == 0:
            print("%s%% finished..."%(100*(step+1)//params["trainStep"]))

        # print(model.linear[0].weight.grad)
        # print(model.linear[0].weight)
        loss.backward()        
        # print(model.linearIn.weight)
        # print(model.linearIn.weight.grad)
        
        # print(model.linear[0].weight)
        # print(model.linear[0].weight.grad)

        # print(model.linear[1].weight)
        # print(model.linear[1].weight.grad)


        # print(model.linearOut.weight)
        # print(model.linearOut.weight.grad)

        
        optimizer.step()        
        scheduler.step()
        
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print (name, param.data.shape)
        #         print (name, param.data)

        # print("---------")


        
def errorFun(output,target,params):
    error = output-target
    error = math.sqrt(torch.mean(error*error))
    # Calculate the L2 norm error.
    ref = math.sqrt(torch.mean(target*target))
    return error/ref   



def test(data,model,device,params):
    output = model(data)
    target = exact(data).to(device)

    error = output-target
    error = math.sqrt(torch.mean(error*error))
    # Calculate the L2 norm error.
    ref = math.sqrt(torch.mean(target*target))
    return error/ref




def exact(data):
    return torch.sin(data)



def main():
    # Parameters
    # torch.manual_seed(21)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["aa"] =  -1
    params["bb"] =   1    
    params["d"] = 1 # 1D
    params["dd"] = 1 # Scalar field
    params["bodyBatch"] = 1 # Batch size
    params["lr"] = 0.01 # Learning rate
    params["width"] = 2 # Width of layers
    params["depth"] = 2 #5 # Depth of the network: depth+2
    params["numQuad"] = 10 # Number of quadrature points for testing
    params["trainStep"] =  10000    
    params["writeStep"] = 50
    params["sampleStep"] = 10
    params["step_size"] = 5000
    params["gamma"] = 0.1
    params["decay"] = 0.0000001

    startTime = time.time()
    model = RitzNet(params).to(device)

    # initWeightAndBias(model)
    print("Generating network costs %s seconds."%(time.time()-startTime))

    optimizer    = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler    = StepLR(optimizer,step_size=params["step_size"],gamma=params["gamma"])
    
    startTime = time.time()

    
    # array = np.zeros([1,1])
    # array[0] = -1;
    # p = torch.from_numpy(array).float().to(device)
    # print('--->', model( p  ))
    

    
    train(model,device,params,optimizer,scheduler)
    print("Training costs %s seconds."%(time.time()-startTime))

    # print(model.parameters())
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data)
    # print("---------")


    model.eval()

    data = testData(params)
    data = torch.from_numpy(data).float().to(device)
    
    testError = test(data,model,device,params)
    print("The test error (of the last model) is %s."%testError)
    torch.save(model.state_dict(),"last_model.pt")

    pltResult(data,model,device,50,params)


    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())


    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

    # for param in model.parameters():
    #     print(param.data)
        

def pltResult(data, model,device,nSample,params):

    target = exact(data)
    output = model(data)
    
    plt.plot(data.detach().cpu().numpy(),output.detach().cpu().numpy(),'ro')
    plt.plot(data.detach().cpu().numpy(),target.detach().cpu().numpy(),'b*')
    
    error = target-output
    error = math.sqrt(torch.mean(error*error)/error.size(dim=1))
    formatted_float = "Error {:.6f}".format(error)

    plt.title(formatted_float)
    plt.show()




if __name__=="__main__":
    main()
