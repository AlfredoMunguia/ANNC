# here is implemented the Higham and Higham problem

import numpy as np 
import matplotlib.pyplot as plt

import math, torch, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.linear1=nn.Linear(2,2)
        self.linear2=nn.Linear(2,3)
        self.linear3=nn.Linear(3,2)


    def forward(self, x):
        x=torch.sigmoid(self.linear1(x))        
        x=torch.sigmoid(self.linear2(x))
        x=torch.sigmoid(self.linear3(x))
        return x


def errorFun(output,target):
    error = output-target
    error = math.sqrt(torch.mean(error*error))
    # Calculate the L2 norm error.
    ref = math.sqrt(torch.mean(target*target))
    return error/ref   


def train(xTrain, yTrain, model,params,optimizer,scheduler):
    model.train()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data   = torch.from_numpy(xTrain).float().to("cpu")
    yTrain = torch.from_numpy(yTrain).float().to("cpu")

    
    for step in range(params["trainStep"]):
        output = model(data)
        model.zero_grad()
        
        
        loss = 0.5*torch.mean( (output-yTrain) * (output-yTrain) )
        error = errorFun(output,yTrain)
        print("Error at Step %s is %s."%(step+1,error))
        
                            
        if 10*(step+1)%params["trainStep"] == 0:
            print("%s%% finished..."%(100*(step+1)//params["trainStep"]))

        loss.backward()        
        optimizer.step()        
        scheduler.step()

def test(model,params):

    x = np.linspace(0,1,params["xnumQuad"])
    y = np.linspace(0,1,params["ynumQuad"])

    xTest,yTest  = np.meshgrid(x,y)
    xxTest       = np.zeros([params["xnumQuad"]*params["ynumQuad"],2])

    k = 0
    for i in range (params["xnumQuad"]):
        for j in range(params["ynumQuad"]):
            xxTest[k] = [xTest[i][j],yTest[i][j]]
            k = k + 1

    data   = torch.from_numpy(np.asarray(xxTest)).float().to("cpu")
    output = model(data)
    output = torch.where(output[:,:] < .54, 0, 1)
    return xTest,yTest,output

    

def main():

    params = dict()
    params["aa"] =  0
    params["bb"] =  1
    params["cc"] =  0
    params["dd"] =  1    

    params["d"] = 2 # 1D
    params["dd"] = 2 # Scalar field
    params["bodyBatch"] = 10 # Batch size
    params["lr"] = 0.01 # Learning rate
    params["width"] = 2 # Width of layers
    params["depth"] = 2 #5 # Depth of the network: depth+2
    params["xnumQuad"] = 20 # Number of quadrature points for testing
    params["ynumQuad"] = 20 # Number of quadrature points for testing
    params["trainStep"] =  10000 #10000    
    params["writeStep"] = 50
    params["sampleStep"] = 10
    params["step_size"] = 5000
    params["gamma"] = 0.1
    params["decay"] = 0.0000001


    xTrain = np.zeros([params["bodyBatch"],2])
    yTrain = np.zeros([params["bodyBatch"],2])
    
    # x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7];
    # x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]

    xTrain[0][0] = 0.1
    xTrain[0][1] = 0.1
    xTrain[1][0] = 0.3
    xTrain[1][1] = 0.4
    xTrain[2][0] = 0.1
    xTrain[2][1] = 0.5
    xTrain[3][0] = 0.6
    xTrain[3][1] = 0.9
    xTrain[4][0] = 0.4
    xTrain[4][1] = 0.2
    xTrain[5][0] = 0.6
    xTrain[5][1] = 0.3
    xTrain[6][0] = 0.5
    xTrain[6][1] = 0.6
    xTrain[7][0] = 0.9
    xTrain[7][1] = 0.2
    xTrain[8][0] = 0.4
    xTrain[8][1] = 0.4
    xTrain[9][0] = 0.7
    xTrain[9][1] = 0.6

    yTrain[0][0] = 1.0
    yTrain[0][1] = 0.0
    yTrain[1][0] = 1.0
    yTrain[1][1] = 0.0
    yTrain[2][0] = 1.0
    yTrain[2][1] = 0.0
    yTrain[3][0] = 1.0
    yTrain[3][1] = 0.0
    yTrain[4][0] = 1.0
    yTrain[4][1] = 0.0
    yTrain[5][0] = 0.0
    yTrain[5][1] = 1.0
    yTrain[6][0] = 0.0
    yTrain[6][1] = 1.0
    yTrain[7][0] = 0.0
    yTrain[7][1] = 1.0
    yTrain[8][0] = 0.0
    yTrain[8][1] = 1.0
    yTrain[9][0] = 0.0
    yTrain[9][1] = 1.0


    
    
    startTime    = time.time()
    model        = Net()
    optimizer    = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler    = StepLR(optimizer,step_size=params["step_size"],gamma=params["gamma"])
    
    startTime    = time.time()

    train(xTrain,yTrain,model,params,optimizer,scheduler)    
    x,y,target = test(model,params)

    # print(xy.shape)
    #print(target[:,0])
    
    plt.scatter(xTrain[:,0],xTrain[:,1],c=yTrain[:,0],s=140,edgecolor="k")
    plt.scatter(x,y,c=target[:,0])

    plt.show()
    
# # Define the Ap and Bp points (Ap are the successful and Bp the failed points)
# Ax=torch.tensor([0.1, 0.3, 0.1, 0.6, 0.4])
# Bx=torch.tensor([0.6, 0.5, 0.9, 0.4, 0.7])
# Ay=torch.tensor([0.1, 0.4, 0.5, 0.9, 0.2])
# By=torch.tensor([0.3, 0.6, 0.2, 0.4, 0.6])
# Ap=torch.stack((Ax,Ay),-1)
# Bp=torch.stack((Bx,By),-1)
# ABp=torch.cat((Ap, Bp),0)
# print(ABp)

# #Define the nodes number in each layer
# nodes_in_layers=torch.tensor([2,2,3,2])

# #making A and B points-plot (it is comment in order to gain clarity)
# plt.scatter(Ax,Ay, s=40,facecolors='none', edgecolors='r')
# plt.plot(Bx,By,"xb",mew=1.2,ms=6)
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.legend(["A category", "B category"])
# plt.show()

        
        

# import torch.optim as optim
# lr=0.05
# net=Net()
# criterion=nn.MSELoss()
# optimizer=optim.SGD(net.parameters(),lr)


# for i in ABp:
#     inputs=i
#     labels=i

#     optimizer.zero_grad()
#     outputs=net(inputs)
#     loss=criterion(outputs,labels)
#     loss.backward()
#     optimizer.step()
    

# print(outputs)
# print(loss)



if __name__=="__main__":
    main()
