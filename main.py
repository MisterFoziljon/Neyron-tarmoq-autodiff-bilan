from sklearn.datasets import load_digits
import numpy as np
import Qatlam as qatlam
import Sozlovchi
import Xatolik
from autodiff import *

def one_hot(n, length):
    label = [0] * length
    label[n - 1] = 1
    return label

mnist = load_digits()
images = np.array([image.flatten() for image in mnist.images])
targets = np.array([one_hot(n, 10) for n in mnist.target])

model = qatlam.Model([
    qatlam.Dense(64),
    qatlam.Dense(32),
    qatlam.Sigmoid(),
    qatlam.Dense(10),
    qatlam.Sigmoid()
])

model.train(images[:100], targets[:100], epochs=10, loss=Xatolik.MSE, optimizer=Sozlovchi.SGD(0.001), batch_size=100)
