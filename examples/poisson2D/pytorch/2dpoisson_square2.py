import numpy as np 
import math, torch, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os


    
def myFun(x,interval):
    if interval[0] <= x  and  x <=  interval[1]:
        return 1.0
    else: return 0.0

    
class DomainRectangle_with_hole(object):

    def __init__(self,params):

        self.a  = params["aa"]
        self.b  = params["bb"]
        self.c  = params["cc"]
        self.d  = params["dd"]
                
        self.nInt    = params["bodyBatch"]
        self.nBdy    = params["bdryBatch"]
        self.nEval   = params["numQuad"]
        
        
    def sampleFromDomain(self):
        array = np.zeros([self.nInt,2])        
    
        for i in range(self.nInt):
            array[i] =  np.random.rand(2)*(self.b-self.a) + self.a
            
        return array

    def getA(self):
        return self.a

    def getB(self):
        return self.b

    def getC(self):
        return self.c

    def getD(self):
        return self.d
    
    
    def sampleFromBoundary(self):

        interval1 = np.array([0.0,       1.0/4.0])
        interval2 = np.array([1.0/4.0,   1.0/2.0])
        interval3 = np.array([1.0/2.0,   3.0/4.0])
        interval4 = np.array([3.0/4.0,   1.0])

        array = np.zeros([self.nBdy,2])

        for i in range(self.nBdy):
            rand0 = np.random.rand()
            rand1 = np.random.rand()

            point1 = np.array([rand1*(self.b-self.a)+self.a,self.c])
            point2 = np.array([rand1*(self.b-self.a)+self.a,self.d])
            point3 = np.array([self.a,rand1*(self.d-self.c)+self.c])
            point4 = np.array([self.b,rand1*(self.d-self.c)+self.c])


            array[i] = myFun(rand0,interval1)*point1 + myFun(rand0,interval2)*point2 + myFun(rand0,interval3)*point3 + myFun(rand0,interval4)*point4
 
        return array

        
    def area(self):
        return (self.b-self.a)*(self.d-self.c)

    def perimeter(self):
        return 2*(self.b-self.a) + 2*(self.d-self.c) 
        



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
        x = torch.tanh(self.linearIn(x)) # Match dimension

        for layer in self.linear:
            x_temp = torch.tanh(layer(x))
            x = x_temp
        
        return self.linearOut(x)


def train(domain,model,device,params,optimizer,scheduler):

    model.train()

    data1 = torch.from_numpy(domain.sampleFromDomain()).float().to(device)
    data2 = torch.from_numpy(domain.sampleFromBoundary()).float().to(device)

    
    x_shift = torch.from_numpy(np.array([params["diff"],0.0])).float().to(device)
    y_shift = torch.from_numpy(np.array([0.0,params["diff"]])).float().to(device)
    data1_x_shift = data1+x_shift
    data1_y_shift = data1+y_shift

    for step in range(params["trainStep"]-params["preStep"]):
        output1 = model(data1)
        output1_x_shift = model(data1_x_shift)
        output1_y_shift = model(data1_y_shift)

        dfdx = (output1_x_shift-output1)/params["diff"] # Use difference to approximate derivatives.
        dfdy = (output1_y_shift-output1)/params["diff"] 

        model.zero_grad()

        # Loss function 1

        fTerm = ffun(data1).to(device)
        loss1 = torch.mean(0.5*(dfdx*dfdx+dfdy*dfdy)-fTerm*output1)*domain.area()

        # Loss function 2
        output2 = model(data2)
        target2 = exact(data2)
        
        loss2 = torch.mean((output2-target2)*(output2-target2) * params["penalty"]*domain.perimeter())
        loss = loss1+loss2              

        if step%params["writeStep"] == params["writeStep"]-1:
            with torch.no_grad():
                target = exact(data1)
                error = errorFun(output1,target,params)
                # print("Loss at Step %s is %s."%(step+params["preStep"]+1,loss.item()))
                print("Error at Step %s is %s."%(step+params["preStep"]+1,error))
            file = open("lossData.txt","a")
            file.write(str(step+params["preStep"]+1)+" "+str(error)+"\n")

        if step%params["sampleStep"] == params["sampleStep"]-1:
            data1 = torch.from_numpy(domain.sampleFromDomain()).float().to(device)
            data2 = torch.from_numpy(domain.sampleFromBoundary()).float().to(device)

            data1_x_shift = data1+x_shift
            data1_y_shift = data1+y_shift

        if 10*(step+1)%params["trainStep"] == 0:
            print("%s%% finished..."%(100*(step+1)//params["trainStep"]))

        loss.backward()

        optimizer.step()
        scheduler.step()

def errorFun(output,target,params):
    error = output-target
    error = math.sqrt(torch.mean(error*error))
    # Calculate the L2 norm error.
    ref = math.sqrt(torch.mean(target*target))
    return error/ref   

def test(domain,model,device,params):
    # numQuad = params["numQuad"]

    data = torch.from_numpy(domain.sampleFromDomain()).float().to(device)
    output = model(data)
    target = exact(data).to(device)

    error = output-target
    error = math.sqrt(torch.mean(error*error))
    # Calculate the L2 norm error.
    ref = math.sqrt(torch.mean(target*target))
    return error/ref

def ffun(data):
    # f = 0.0
    # return 0.0*torch.ones([data.shape[0],1],dtype=torch.float)

    #f = -(-108*(cos(5.4*y)+1.25)/(6*(3*x-1)**2+6)**2 + 2592*(3*x-1)**2*(cos(5.4*y)+1.25)/(6*(3*x-1)^2+6)**3 -29.16*cos(5.4*y)/(6*(3*x-1)**2+6)

    output =-(-108*(torch.cos(5.4*data[:,1])+1.25)/ ((6*(3*data[:,0]-1)**2+6)**2) \
             +(2592*(3*data[:,0]-1)**2*(torch.cos(5.4*data[:,1])+1.25))/((6*(3*data[:,0]-1)**2+6)**3) \
             -29.16*torch.cos(5.4*data[:,1])/(6*(3*data[:,0]-1)**2+6))
    
    return  torch.reshape(output,(data.shape[0],1))

    #f = 2*y**2+2*x**2
    #return -(2*data[:,1]**2+2*data[:,0]**2)

    #f = 6*x*y*(1-y)-2*x**3
    #return 6*data[:,0]*data[:,1]*(1-data[:,1])-2*data[:,0]**3


def exact(data):
    # f = 0 ==> u = xy
    # output = data[:,0]*data[:,1]
    # return output.unsqueeze(1)

    output = (  1.25 + torch.cos(5.4*data[:,1])  )  / ( 6 + 6*(3*data[:,0]-1)**2 )
    return torch.reshape(output,(data.shape[0],1))

    #output = data[:,0]**2*data[:,1]**2
    #return output

    #f = y(1-y)x**3
    # output = data[:,1]*(1-data[:,1])*data[:,0]**3
    # return output


def solution(x,y):
    # return x*y
    return (1.25+math.cos(5.4*y))/(6+6*(3*x-1)**2)    
    #return x**2*y**2
    #return y*(1-y)*x**3

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# def rough(r,data):
#     output = r**2-r*torch.sum(data*data,dim=1)**0.5
#     return output.unsqueeze(1)

def main():
    # Parameters
    # torch.manual_seed(21)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["aa"] =  0
    params["bb"] =  1
    params["cc"] =  0
    params["dd"] =  1
    
    params["d"] = 2 # 2D
    params["dd"] = 1 # Scalar field
    params["bodyBatch"] = 1024 # Batch size
    params["bdryBatch"] = 1024 # Batch size for the boundary integral
    params["lr"] = 0.01 # Learning rate
    params["preLr"] = 0.01 # Learning rate (Pre-training)
    params["width"] = 12 # Width of layers
    params["depth"] = 8 # Depth of the network: depth+2
    params["numQuad"] = 40000 # Number of quadrature points for testing
    params["trainStep"] =  50000
    params["penalty"] = 500
    params["preStep"] = 0
    params["diff"] = 0.001
    params["writeStep"] = 50
    params["sampleStep"] = 10
    params["step_size"] = 5000
    params["gamma"] = 0.3
    params["decay"] = 0.00001

    startTime = time.time()
    model = RitzNet(params).to(device)
    print("Generating network costs %s seconds."%(time.time()-startTime))

    preOptimizer = torch.optim.Adam(model.parameters(),lr=params["preLr"])
    optimizer    = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler    = StepLR(optimizer,step_size=params["step_size"],gamma=params["gamma"])
    domain       = DomainRectangle_with_hole(params)
    
    startTime = time.time()
    # preTrain(model,device,params,preOptimizer,None,exact)
    train(domain,model,device,params,optimizer,scheduler)
    print("Training costs %s seconds."%(time.time()-startTime))

    model.eval()
    testError = test(domain,model,device,params)
    print("The test error (of the last model) is %s."%testError)
    print("The number of parameters is %s,"%count_parameters(model))

    torch.save(model.state_dict(),"last_model.pt")

    pltResult(domain,model,device,50,params)

def pltResult(domain,model,device,nSample,params):

    a = domain.getA()
    b = domain.getB()
    c = domain.getC()
    d = domain.getD()
    
    xList = np.linspace(a,b,nSample)
    yList = np.linspace(c,d,nSample)

    xx = np.zeros([nSample,nSample])
    yy = np.zeros([nSample,nSample])
    zz = np.zeros([nSample,nSample])
    zze = np.zeros([nSample,nSample])
    for i in range(nSample):
        for j in range(nSample):
            xx[i,j] = xList[i]
            yy[i,j] = yList[j]
            coord = np.array([xx[i,j],yy[i,j]])
            zz[i,j]  = model(torch.from_numpy(coord).float().to(device)).item()
            zze[i,j] = solution(xx[i,j],yy[i,j]) # Plot the exact solution.

    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.plot_wireframe(xx, yy, zz, color='black')
    scatter1 = ax.scatter(xx,yy,zz,'o',color='blue')
    scatter2 = ax.scatter(xx,yy,zze,'*',color='red')
    ax.legend([scatter1,scatter2],['approx','exact'])

    approx = torch.from_numpy(zz).float().to(device)
    exact  = torch.from_numpy(zze).float().to(device)
    
    error = approx-exact
    error = math.sqrt(torch.mean(error*error)*domain.area())

    formatted_float = "Error L2 {:.6f}".format(error)
    ax.set_title(formatted_float,fontsize = 10);
    print(error)
    
    plt.show()


    # file = open("nSample.txt","w")
    # file.write(str(nSample))

    # file = open("Data.txt","w")
    # writeSolution.write(xx,yy,zz,nSample,file)

    # edgeList2 = [[0.3*math.cos(i)+0.3,0.3*math.sin(i)] for i in thetaList]
    # edgeList1 = [[-1.0,-1.0],[1.0,-1.0],[1.0,1.0],[-1.0,1.0],[-1.0,-1.0]]
    # writeSolution.writeBoundary(edgeList1,edgeList2)

if __name__=="__main__":
    main()
