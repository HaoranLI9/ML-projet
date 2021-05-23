import numpy as np
from Loss import *
class Sequentiel:
    def __init__(self,*args):
        self.modules = []
        for module in args:
            self.modules.append(module)

    def forward(self, input):
        inputs = [input]
        for module in self.modules:
            inputs.append(module.forward(input))
            input = inputs[-1]
        
        return inputs

    def backward(self, outputs, lastDelta, eps):
        tmpDelta = lastDelta

        for i in range(len(outputs)-2,-1, -1):

            module = self.modules[i]

            #print(module, 'delta',tmpDelta.shape)
            #print(datax.shape, tmpDelta.shape)
            delta = module.backward_delta(outputs[i], tmpDelta)

            #print(module, 'delta min(), max()', delta.min(), delta.max())

            module.backward_update_gradient(outputs[i], tmpDelta)

            module.update_parameters(gradient_step = eps)
            
            module.zero_grad()

            tmpDelta = delta



class Optim:
    def __init__(self,net,loss,eps):
        self.net = net
        self.loss = loss
        self.eps = eps
    
    def step(self,batch_x,batch_y):
        
        outputs = self.net.forward(batch_x)

        # print('yhat after softmax()', yhat.shape,yhat)
        #print(outputs)
        loss = self.loss.forward(batch_y,outputs[-1])

        #print("sum loss:",loss.sum())
        
        lastDelta = self.loss.backward(batch_y,outputs[-1])

        #print('sum lastDelta', lastDelta.sum())

        self.net.backward(outputs, lastDelta, self.eps)

        return loss.sum()

        

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def mini_SGD(seq, alltrainx, alltrainy ,batch_size = 100, nb_iteration = 300, loss_fonction = Softmax_CELoss(), eps = 1e-5):
    
    sum_loss = []
    N = alltrainx.shape[0]
    N_epoch = int(N/batch_size) # batch_size = 1-> GD, batch_size = N -> SGD
    
    opt = Optim(seq, loss_fonction, eps)
    for it in range(nb_iteration):
        loss = []
        for epch in range(N_epoch):
            idx = np.random.choice(range(N), batch_size)
            x, y = alltrainx[idx], alltrainy[idx]
            tmp_loss = opt.step(x,y)
            loss.append(tmp_loss)
        sum_loss.append(np.mean(loss))
        if it % 20 == 0:
            print("iteration", it, "loss =", np.mean(loss)) 
                       
    #plt.plot(range(len(loss)),loss)
    #plt.savefig('seq loss curve bceloss.png')
    #plt.show()

         
    return seq, sum_loss
