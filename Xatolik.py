import numpy as np
import autodiff as ad

def MSE(pred, real):
    loss = ad.reduce_mean((pred - real)**2)
    return loss

def crossentropy(pred, real):
    loss = -1 * ad.reduce_mean(real * ad.log(pred))
    return loss

def binary_crossentropy(pred, real):
    loss = -1 * ad.reduce_mean(real * ad.log(pred)+(1-real)*ad.log(1-pred))
    return loss
