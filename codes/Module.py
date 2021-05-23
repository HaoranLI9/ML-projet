import numpy as np

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        self._gradient = 0

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        
        #print(self,'parameters -=', (gradient_step*self._gradient).sum() )

        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass

    def __str__(self):
        return self.__class__.__name__

####################################
#########  LINEAR ##################
####################################
class Linear(Module):
    def __init__(self, input, output):
        '''
        param
        ------
        _parameters: (input,output)
        _gradient  : (input,output)
        '''
        self._parameters = 2 *(np.random.rand(input,output)-0.5)
        self._gradient = np.zeros((input,output))

    def zero_grad(self):
        ## Annule gradient
        self._gradient = np.zeros(self._gradient.shape)
        

    def forward(self, X):
        ## Calcule la passe forward
        '''
        param
        ------
        X : (batch,input)
        return
        ------
        output of layer : (batch,output)
        '''
        
        return X@self._parameters 
        
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        '''
        param:
        ------
        input: Z^h, (batch,input)
        delta: gradient of Loss with Z, (batch,output)

        update self._gradient which is (input,output)
        by the gradient of z^h with respect of w multiplied by delta
        '''
        assert input.shape[0] == delta.shape[0]

        #print(input.shape, self._gradient.shape)
        assert input.shape[1] == self._gradient.shape[0]
        assert delta.shape[1] == self._gradient.shape[1]

        self._gradient += input.T@delta


    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        '''
        param
        ------
        delta^h: next gradient, (batch,output)
        input: Z^(h-1), (batch,input)

        return
        ------
        delta^(h-1) for previous layer: (batch,input)
        '''
        assert input.shape[1] == self._parameters.shape[0]
        assert delta.shape[1] == self._parameters.shape[1]
        assert input.shape[0] == delta.shape[0]
        
        return delta@(self._parameters.T)

#################################################
############ Activation Function ################
#################################################
class TanH(Module):      

    def forward(self, x):
        #print(x.max(), x.min())
        #ex, e_x = np.exp(x), np.exp(-x)
        #return (ex - e_x) / (ex + e_x)
        return np.tanh(x)

    def update_parameters(self, gradient_step=1e-3):
        '''
        activation function has no parameter
        '''        
        pass    
    
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        '''
        activation function has no gradient Z^h with respect of w
        '''
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        '''
        calculate alpha^(h-1)
        param
        ------
        for activation functions, output==input
        delta^h: next gradient, (batch,output)
        input: Z^(h-1), (batch,input)

        return
        ------
        delta^(h-1) for previous layer: (batch,input)
        '''
        assert input.shape == delta.shape

        # derivative of tanh(x) is 1-tanh(x)**2
        
        return (1-self.forward(input)**2)*delta


class Sigmoid(Module):

    def forward(self, X):
        ## Calcule la passe forward
        '''
        param
        ------
        - X  :  (N, *)
        - Output : (N, *), same shape as the input
        '''
        return np.exp(np.fmin(X, 0)) / (1 + np.exp(-np.abs(X)))

    def update_parameters(self, gradient_step=1e-3):
        '''
        activation function has no parameter
        '''        
        pass

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        '''
        activation function has no gradient Z^h with respect of w
        '''
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        '''
        calculate alpha^(h-1)
        param
        ------
        for activation functions, output==input
        delta^h: next gradient, (batch,output)
        input: Z^(h-1), (batch,input)

        return
        ------
        delta^(h-1) for previous layer: (batch,input)
        '''
        assert input.shape == delta.shape
        # derivative of sigmoid(x) is sigmoid(x)(1-sigmoid(x))
        sig = self.forward(input)
        return (sig*(1-sig))*delta



class ReLU(Module):
    def forward(self, x):
        
        return np.maximum(x, 0)

    def update_parameters(self, gradient_step=1e-3):
        '''
        activation function has no parameter
        '''        
        pass

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        '''
        activation function has no gradient Z^h with respect of w
        '''
        pass

    def backward_delta(self, input, delta):
        '''
        calculate alpha^(h-1)
        param
        ------
        for activation functions, output==input
        delta^h: next gradient, (batch,output)
        input: Z^(h-1), (batch,input)

        return
        ------
        delta^(h-1) for previous layer: (batch,input)
        '''
        assert input.shape == delta.shape         
        return (np.where(self.forward(input)>0, 1, 0))*delta


class Softmax(Module):

    def forward(self, x):
        """
        if len(x.shape)>1:
            x = x - np.max(x)
            exp_x = np.exp(x)
            sum =  exp_x.sum(axis = 1).reshape(-1,1)
            #print(sum)
            softmax_x = exp_x / sum
            return softmax_x
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place

         """
        orig_shape = x.shape

        if len(x.shape) > 1:
            # Matrix
            exp_minmax = lambda x: np.exp(x - np.max(x))
            denom = lambda x: 1.0 / np.sum(x)
            x = np.apply_along_axis(exp_minmax,1,x)
            denominator = np.apply_along_axis(denom,1,x) 

            if len(denominator.shape) == 1:
                denominator = denominator.reshape((denominator.shape[0],1))

            x = x * denominator
        else:
            # Vector
            x_max = np.max(x)
            x = x - x_max
            numerator = np.exp(x)
            denominator =  1.0 / np.sum(numerator)
            x = numerator.dot(denominator)

        assert x.shape == orig_shape
        return x

    def update_parameters(self, gradient_step=1e-3):
        '''
        activation function has no parameter
        '''        
        pass

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        '''
        activation function has no gradient Z^h with respect of w
        '''
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        '''
        calculate delta^(h-1)
        gradient of softmax(x_i) with respect of x_j is 
        softmax(x_i)*(1 - softmax(x_j)) when i==j, and -softmax(x_i)*softmax(x_j) when i!=j
        
        param
        ------
        for activation functions, output==input
        delta^h: next gradient, (batch,output)
        input: Z^(h-1), (batch,input)
        return
        ------
        delta^(h-1) for previous layer: (batch,input)
        
        assert input.shape == delta.shape
        f = self.forward(input)
        return f*(1 - f)*delta
        '''

        ##Normalement, on l'utilise avec CELoss pour le problème de multiclasse, donc on utilise pas backward ici.
        raise NotImplementedError

#################################################
################# CNN ###########################
#################################################

class Conv1D(Module):
    def __init__(self,k_size,chan_in,chan_out,stride):
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride

        np.random.seed(0)
        self._parameters =2*(np.random.random((k_size,chan_in,chan_out))-0.5) * 1e-1 # dim (k_size, chan_in, chan_out)
        self._gradient = np.zeros((k_size,chan_in,chan_out)) # dim : k_size, chan_in, chan_out

    def forward(self,X):
        '''
        params:
        -------
        X : dim (batch,length,chan_in)

        return:
        -------
        dim (batch,(length-k_size)/stride + 1, chan_out)
        '''
        b,l,cin = X.shape

        res = np.zeros((b, (l-self.k_size)//self.stride + 1, self.chan_out))
        for n in range(b):
            X0 = X[n] # dim (length,chan_in)
            for f in range(self.chan_out): # for every filter
                W = self._parameters[:,:,f] # dim (k_size,chan_in)

                for i in range(0,res.shape[1]):
                    j = i*self.stride
                    X1 = X0[j:j+self.k_size] # dim (k_size,chan_in)
                    
                    res[n,i,f] = (X1 * W).sum()

        return res

    def backward_update_gradient(self, input, delta):
        '''
        params:
        -------
        input : dim (batch,length,chan_in)
        delta : dim (batch,(length-k_size)/stride + 1,chan_out)
        self._gradient : (k_size,chan_in,chan_out)
        
        return:
        -------
        None
        '''
        assert input.shape[2] == self.chan_in
        assert delta.shape[2] == self.chan_out
        assert delta.shape[1] == (input.shape[1]-self.k_size)//self.stride + 1
        assert delta.shape[0] == input.shape[0]
        b, length_out, chan_out = delta.shape
        #g = np.zeros((self.k_size,self.chan_in,self.chan_out))
        for n in range(b):
            X0 = input[n] # dim (length,chan_in)
            for z in range(chan_out):
                for i in range(length_out):
                    Xs = X0[i:i+self.k_size] # dim (k_size, chan_in)
                                             # derivative of o_i with respect of w is x
                    delta0 = delta[n,i,z]
                    self._gradient[:,:,z] += Xs*delta0
                    #g[:,:,z] += Xs*delta0
        #return g
    def update_parameters(self, gradient_step=1e-5):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass

    def backward_delta(self, input, delta):
        '''
        params:
        -------
        input : dim (batch,length,chan_in)
        delta : dim (batch,(length-k_size)/stride + 1,chan_out)
        self._parameters : dim (k_size, chan_in, chan_out)
        
        return:
        -------
        delta : dim(batch, length, chan_in)        
        '''    

        b, length_out, chan_out = delta.shape
        
        res = np.zeros(input.shape)
        res = np.array(res, dtype=np.float64)

        for n in range(b):
            X0 = input[n] # dim (length,chan_in)
            for z in range(chan_out):
                Ws = self._parameters[:,:,z] # dim (k_size, chan_in)
                                             # derivative of o_i with respect of x is w
                for i in range(length_out):
                    delta0 = delta[n,i,z]
                    res[n,i:i+self.k_size,:] += Ws*delta0
        return res
    
    def zero_grad(self):
        self._gradient = np.zeros(self._gradient.shape)
'''
a = np.array([[[3,4,8],[5,6,7],[3,3,9],[6,7,1]]])
b = np.array([[[1,2,3,6],[4,2,6,10],[10,11,12,5]]])
c1 = Conv1D(2,3,4,1)
#f = c1.forward(a)
print('c1',c1.backward_delta(a,b))
print('c1',c1.backward_update_gradient(a,b))

c2 = Conv1D2(2,3,4,1)
print('c2',c2.backward_delta(a,b))
print('c2',c2.backward_update_gradient(a,b))
'''
#%%
class MaxPool1D(Module):
    def __init__(self,k_size,stride):
        self.k_size = k_size
        self.stride = stride
        self.maxind = None
    
    def forward(self,X):
        '''
        params:
        -------
        X : dim (batch,length,chan_in)

        return:
        -------
        dim (batch,(length-k_size)/stride + 1, chan_in)
        '''
        b,l,cin = X.shape

        res = np.zeros((b, (l-self.k_size)//self.stride + 1, cin))
        self.maxind = np.zeros(res.shape)

        for i in range(0, res.shape[1]):
            
            self.maxind[:,i] = np.argmax(X[:, (i * self.stride): (i * self.stride + self.k_size)],
                                                 axis=1) + i * self.stride
            res[:,i,:] = np.max(X[:,(i*self.stride) : (i*self.stride + self.k_size)], axis=1)
        self.maxind = self.maxind.astype(int)
        return res

    def update_parameters(self, gradient_step=1e-5):
        '''
        no parameter
        '''        
        pass
        

    def backward_update_gradient(self, input, delta):
        '''
        MaxPool1D has no parameter.
        '''        
        pass

    def backward_delta(self, input, delta):
        '''
        There is no gradient with respect to non maximum values, since changing them slightly does not affect the output. 
        Further, the max is locally linear with slope 1, with respect to the input that actually achieves the max. 
        Thus, the gradient from the next layer is passed back to only that neuron which achieved the max. 
        All other neurons get zero gradient.

        chan_in == chan_out
        
        param:
        -------
        input: dim (batch,length,chan_in)
        delta: dim (batch,(length-k_size)//stride + 1,chan_in)
        
        return:
        -------
        delta, dim (batch,length,chan_in)
        '''
        b,l,c = input.shape        
        res = np.zeros((b,l,c))
        
        for n in range(b):
            X0 = input[n]
            ind = self.maxind[n]
            #print(ind)
            for i in range(ind.shape[0]):
                for j in range(ind.shape[1]):
                    res[n,ind[i,j],j] = delta[n,i,j]
        
        return res
'''
a = np.array([[[3,4,8],[5,6,7],[3,3,9],[6,7,1]]])
b = np.array([[[1,2,3],[4,2,6],[10,11,12]]])
m1 = MaxPool1D(2,1)
m1.forward(a)
m1.backward_delta(a, b)
'''
#%%


class Flatten(Module):
    def __init__(self):
        pass
    
    def forward(self,X):
        '''
        params:
        -------
        X : dim (batch,length,chan_in)

        return:
        -------
        dim (batch,length*chan_in)
        '''
        return X.reshape((len(X),-1))
        
    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        '''
        param:
        ------
        input : resPool, dim (batch,length,chan_in)
        delta : delta of lin, dim (batch,length*chan_out)
        
        return:
        dim (batch,length,chan_in)
        '''
        return delta.reshape(input.shape)

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def update_parameters(self, gradient_step=1e-5):
        '''
        no parameter
        '''        
        pass    