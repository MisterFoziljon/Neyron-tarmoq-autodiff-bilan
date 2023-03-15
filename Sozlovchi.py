class SGD:
    def __init__(self, lr):
        self.lr = lr

    def delta(self, param):
        return param.gradient * self.lr

    def __call__(self, model, loss):
        from Qatlam import Layer
        loss.get_gradients()

        for layer in model.layers:
            if isinstance(layer, Layer):
                layer.update(self)
