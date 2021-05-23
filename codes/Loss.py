import numpy as np
from Module import Softmax

class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

class MSELoss(Loss):
    def forward(self, y, yhat):
        '''
        param
        ------
        y    : (batch,d)
        yhat : (batch,d)

        return
        ------
        loss L: (batch,1)
        '''
        assert y.shape == yhat.shape
        return np.linalg.norm((y-yhat),axis=1)**2 

    def backward(self, y, yhat):
        '''
        return
        ------
        gradient: derivative of loss with yhat
        '''
        assert y.shape == yhat.shape
        return 2*(yhat-y)

class BCELoss(Loss):
    def forward(self, y, yhat):
        '''
        param
        ------
        y    : (batch,d)
        yhat : (batch,d)

        return
        ------
        loss L: (batch,1)
        
        '''
        assert y.shape == yhat.shape
        res = np.sum(-(y* np.maximum(np.log(yhat+np.e**-100), -100) + (1-y)* np.maximum(-100, np.log(1-yhat+np.e**-100))), axis = 1)
        return res

    def backward(self, y, yhat):
        '''
        y    : (batch,d)
        yhat : (batch,d)
        return gradient of loss with yhat
        formula : -(y_i/yhat) + ( (1-y_i) / (1-yhat))
        '''
        assert y.shape == yhat.shape
        return -y/(yhat+np.e**-100) + ( (1-y)/(1-yhat+np.e**-100) )
        
class Softmax_CELoss(Loss):
    '''
    cross entropy loss
    '''
    def forward(self, y, yhat):
        '''
        param
        ------
        y    : (batch,d)
        yhat : (batch,d)

        return
        ------
        loss L: (batch,1)
        
        '''
        assert y.shape == yhat.shape
        yhat = Softmax().forward(yhat)
        return np.sum(-(y*np.log(yhat)),axis=1)

    def backward(self, y, yhat):
        '''
        gradient of loss with respect of yhat
        we choose index for yhat from y, because there is only one concerned value
        by defaut, softmax takes z_i as input and produces multiclass probabilities
        q_y = yhat[np.nonzero(y)]
        Loss = -log(q_y) = -log(softmax(z_y))

        => gradient of loss with resepect of z_i = q_i - f, f = 1 if i == y, 0 otherwise
        
        cf http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html
        '''
        assert y.shape == yhat.shape

        ind = np.nonzero(y) # shape (batch,1)
        yhat[ind] -= 1
        return yhat