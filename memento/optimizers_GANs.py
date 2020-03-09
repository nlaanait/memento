import torch
import numpy
from torch import Tensor
from torch import autograd
from torch.autograd import Variable
from torch.autograd import grad
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.parallel


class CGD_Jacobi():
    r"""Implements stochastic gradient descent (optionally with momentum).
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, lr=1.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.lr = lr

    def step(self, f, g, model_x, model_y):

        grad_f_x = autograd.grad(f, model_x.main.parameters(), create_graph=True, allow_unused=True)
        grad_f_y = autograd.grad(f, model_y.main.parameters(), create_graph=True, allow_unused=True)
        grad_g_x = autograd.grad(g, model_x.main.parameters(), create_graph=True, allow_unused=True)
        grad_g_y = autograd.grad(g, model_y.main.parameters(), create_graph=True, allow_unused=True)

        hess_f_xy = hv(grad_f_y, model_x)
 
        hess_f_xy_x = []
        for i, j in zip( hess_f_xy, grad_g_x ):
            hess_f_xy_x.append( i * j )            

        for param_cur, param_update1, param_update2 in zip(model_x.main.parameters(), grad_f_x, hess_f_xy_x):
            param_cur.data -= (self.lr * param_update1.data)
            param_cur.data -= (self.lr * param_update2.data)

def hv(grad, model):
        
        v = []

        for p in grad:
            v.append(torch.ones_like(p))

        v = tuple(i for i in v) 
        
        Hv = autograd.grad(grad, model.main.parameters(), grad_outputs=v, retain_graph=False, allow_unused=True)

        return Hv


# Setting hyperparameters
batchSize = 64 
imageSize = 64

# Creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2)

def weights_init(m, verbose=0):
    classname = m.__class__.__name__
    if(verbose==1):
        print(classname)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
        
class G(nn.Module):
   def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            nn.Tanh()
        )
        
   def forward(self, input):
        output = self.main(input)
        return output        
        

class D(nn.Module):
   def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

   def forward(self, input):
        output = self.main(input)
        return output.view(-1)        
        
    
netG = G() 
netG.apply(weights_init)   
netD = D() 
netD.apply(weights_init) 
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizer = CGD_Jacobi(lr = 0.0002)
    


for epoch in range(25):
    for i, data in enumerate(dataloader, 0):

        netD.zero_grad()
        
        # Train discriminator on real data
        real, _ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0]))
        output = netD(input)
        errD_real = criterion(output, target)
        
        # Train discriminator on fake data
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = netD(fake)
        errD_fake = criterion(output, target)
        
        # Backward propagate the update for the weights 
        errD = errD_real + errD_fake
        errD.backward(retain_graph=True)
        optimizer.step(errD, errG, netD, netG)
        #optimizerD.step()
         
        # Next step is to update the weights of the neural network of the generator
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = netD(fake)
        errG = criterion(output, target)
        errG.backward(retain_graph=True)
        #optimizer.step(errG, errD, netG, netD)
        optimizerG.step()
  
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data.item(), errG.data.item()))
