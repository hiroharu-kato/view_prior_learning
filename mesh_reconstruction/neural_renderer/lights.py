import chainer

class Light:
    pass


class DirectionalLight(Light):
    def __init__(self, color, direction, backside=False):
        self.color = color
        self.direction = direction
        self.backside = backside


class AmbientLight(Light):
    def __init__(self, color):
        self.color = color


class SpecularLight(Light):
    def __init__(self, color, alpha=None, backside=False):
        self.color = color
        self.backside = backside
        if alpha is not None:
            self.alpha = alpha
        else:
            xp = chainer.cuda.get_array_module(color)
            self.alpha = xp.ones(color.shape[0], 'float32')