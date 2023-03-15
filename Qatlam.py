import autodiff as ad
import numpy as np
import Xatolik 
import Sozlovchi
from tqdm import tqdm

np.random.seed(345)

class Layer:
    def __init__(self):
        pass

class Dense(Layer):
    def __init__(self, units):
        self.units = units
        self.w = None
        self.b = None

    #Forward propagation for 1 hidden layer
    def __call__(self, x): 
        if self.w is None:
            self.w = ad.Tensor(np.random.uniform(size=(x.shape[-1], self.units), low=-1/np.sqrt(x.shape[-1]), high=1/np.sqrt(x.shape[-1])))
            self.b = ad.Tensor(np.zeros((1, self.units)))
        return x @ self.w + self.b

    #Gradient Descent
    def update(self, optim): 
        self.w.value -= optim.delta(self.w)
        self.b.value -= optim.delta(self.b)

        self.w.grads = []
        self.w.dependencies = []
        self.b.grads = []
        self.b.dependencies = []

class Sigmoid:
    def __call__(self, x):
        return np.e**x /(1 + np.e**x)

class Softmax:
    def __call__(self, x):
        s = np.array(x).reshape(-1,1)
        return ad.Tensor(np.diagflat(s) - np.dot(s, s.T))
    
class Relu:
    def __call__(self, x):
        self.output = np.maximum(0, x)
        return self.output
        
class Model:
    def __init__(self, layers):
        self.layers = layers

    #Forward propagation
    def __call__(self, x): 
        output = x

        for layer in self.layers:
            output = layer(output)

        return output

    def train(self, x, y, epochs=10, loss = Xatolik.MSE, optimizer=Sozlovchi.SGD(lr=0.1), batch_size=1):
        for epoch in range(epochs):
            LOSS = 0
            print (f"EPOCH", epoch + 1)
            for batch in tqdm(range(0, len(x), batch_size)):
                output = self(x[batch:batch+batch_size])
                l = loss(output, y[batch:batch+batch_size])
                optimizer(self, l)
                LOSS += l
            
            print ("LOSS", LOSS.value)
            print (" ")
