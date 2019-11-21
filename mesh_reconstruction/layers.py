import string

import chainer


class Padding(chainer.Function):
    def __init__(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def forward_cpu(self, inputs):
        return self.forward(inputs)

    def forward_gpu(self, inputs):
        return self.forward(inputs)

    def backward_cpu(self, inputs, grad_outputs):
        return self.backward(inputs, grad_outputs)

    def backward_gpu(self, inputs, grad_outputs):
        return self.backward(inputs, grad_outputs)

    def forward(self, inputs):
        x = inputs[0]
        bs, nc, h, w = x.shape
        xp = chainer.cuda.get_array_module(x)
        y = xp.zeros((bs, nc, h + self.top + self.bottom, w + self.left + self.right), x.dtype)
        y[:, :, self.top:self.top + h, self.left:self.left + w] = x.copy()
        return y,

    def backward(self, inputs, gradients):
        x = inputs[0]
        gradient_in = gradients[0]
        bs, nc, h, w = x.shape
        gradient_out = gradient_in[:, :, self.top:self.top + h, self.left:self.left + w]
        return gradient_out,


class RandomCropping(chainer.Function):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.mys = None
        self.mxs = None

    def forward_cpu(self, inputs):
        raise NotImplementedError

    def backward_cpu(self, inputs, grad_outputs):
        raise NotImplementedError

    def forward_gpu(self, inputs):
        images_in = inputs[0]
        bs, nc, height_in, width_in = images_in.shape
        xp = chainer.cuda.get_array_module(images_in)
        images_out = xp.zeros((bs, nc, self.height, self.width), 'float32')

        if self.mys is None:
            mys = xp.random.randint(height_in - self.height + 1, size=bs).astype('int32')
            mxs = xp.random.randint(width_in - self.width + 1, size=bs).astype('int32')
            self.mys = mys
            self.mxs = mxs

        chainer.cuda.elementwise(
            'raw float32 pixel_in, raw int32 mys, raw int32 mxs',
            'float32 pixel_out',
            string.Template('''
                int bn = i / (${num_channels} * ${height_out} * ${width_out});
                int x = i % ${width_out};
                int y = (i / ${width_out}) % ${height_out};
                int cn = (i / (${width_out} * ${height_out})) % ${num_channels};
                int my = mys[bn];
                int mx = mxs[bn];
                pixel_out = pixel_in[
                    bn * ${num_channels} * ${height_in} * ${width_in} + 
                    cn * ${height_in} * ${width_in} + 
                    (y + my) * ${width_in} +
                    (x + mx)
                ];
            ''').substitute(
                num_channels=nc,
                height_in=height_in,
                width_in=width_in,
                height_out=self.height,
                width_out=self.width,
            ),
            'function',
        )(images_in, self.mys, self.mxs, images_out)
        return images_out,

    def backward_gpu(self, inputs, gradients):
        images_in = inputs[0]
        gradient_in = gradients[0]
        bs, nc, height_in, width_in = images_in.shape
        xp = chainer.cuda.get_array_module(images_in)
        gradient_out = xp.zeros(images_in.shape, 'float32')

        chainer.cuda.elementwise(
            'raw float32 pixel_out, raw int32 mys, raw int32 mxs',
            'float32 pixel_in',
            string.Template('''
                int bn = i / (${num_channels} * ${height_in} * ${width_in});
                int cn = (i / (${width_in} * ${height_in})) % ${num_channels};
                int y = (i / ${width_in}) % ${height_in};
                int x = i % ${width_in};
                int my = mys[bn];
                int mx = mxs[bn];
                if (y - my < 0) return;
                if (x - mx < 0) return;
                if (${height_out} <= y - my) return;
                if (${width_out} <= x - mx) return;
                pixel_in = pixel_out[
                    bn * ${num_channels} * ${height_out} * ${width_out} + 
                    cn * ${height_out} * ${width_out} + 
                    (y - my) * ${width_out} +
                    (x - mx)
                ];
            ''').substitute(
                num_channels=nc,
                height_in=height_in,
                width_in=width_in,
                height_out=self.height,
                width_out=self.width,
            ),
            'function',
        )(gradient_in, self.mys, self.mxs, gradient_out)
        return gradient_out,


class InvertGradient(chainer.Function):
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def forward_cpu(self, inputs):
        return self.forward(inputs)

    def forward_gpu(self, inputs):
        return self.forward(inputs)

    def backward_cpu(self, inputs, grad_outputs):
        return self.backward(inputs, grad_outputs)

    def backward_gpu(self, inputs, grad_outputs):
        return self.backward(inputs, grad_outputs)

    def forward(self, inputs):
        return inputs

    def backward(self, inputs, gradients):
        return [-g * self.lambda_ for g in gradients]


class DummyLayer(chainer.Function):
    def __init__(self, *args):
        super(DummyLayer, self).__init__()

    def forward_cpu(self, inputs):
        return self.forward(inputs)

    def forward_gpu(self, inputs):
        return self.forward(inputs)

    def backward_cpu(self, inputs, grad_outputs):
        return self.backward(inputs, grad_outputs)

    def backward_gpu(self, inputs, grad_outputs):
        return self.backward(inputs, grad_outputs)

    def forward(self, inputs):
        return inputs

    def backward(self, inputs, gradients):
        return gradients


def padding(images, top, bottom, left, right):
    return Padding(top, bottom, left, right)(images)


def random_cropping(images, height, width):
    return RandomCropping(height, width)(images)


def invert_gradient(data, lambda_):
    return InvertGradient(lambda_)(data)
