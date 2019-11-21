import chainer
import chainer.functions as cf
import chainer.links as cl

import sn.sn_convolution_2d
import sn.sn_linear


def to_one_hot_vector(data, num_classes):
    xp = chainer.cuda.get_array_module(data)
    data2 = xp.zeros((data.size, num_classes), 'float32')
    data2[xp.arange(data.size), data] = 1
    return data2


class BasicDiscriminator(chainer.Chain):
    def __init__(self, dim_in=4):
        super(BasicDiscriminator, self).__init__()
        with self.init_scope():
            dim_hidden = [128, 256, 512, 1024]
            kernel_size = 5
            stride = 2
            pad = 2
            shape_after_conv = 4

            init = chainer.initializers.HeNormal()
            self.conv1 = cl.Convolution2D(dim_in, dim_hidden[0], kernel_size, stride, pad, initialW=init)
            self.conv2 = cl.Convolution2D(dim_hidden[0] * 2, dim_hidden[1], kernel_size, stride, pad, initialW=init)
            self.conv3 = cl.Convolution2D(dim_hidden[1], dim_hidden[2], kernel_size, stride, pad, initialW=init)
            self.conv4 = cl.Convolution2D(dim_hidden[2], dim_hidden[3], kernel_size, stride, pad, initialW=init)
            self.linear_out = cl.Linear(dim_hidden[3] * shape_after_conv * shape_after_conv, 1, initialW=init)

    def __call__(self, x):
        batch_size = x.shape[0]
        if x.shape[2] != 64:
            x = cf.resize_images(x, (64, 64))
        h = cf.leaky_relu(self.conv1(x))  # [32 -> 16]
        h = cf.leaky_relu(self.conv2(h))  # [32 -> 16]
        h = cf.leaky_relu(self.conv3(h))  # [16 -> 8]
        h = cf.leaky_relu(self.conv4(h))  # [8 -> 4]
        h = cf.reshape(h, (batch_size, -1))
        y = self.linear_out(h)
        return y


class PascalBasicDiscriminator(chainer.Chain):
    def __init__(self, dim_in=4):
        super(PascalBasicDiscriminator, self).__init__()
        with self.init_scope():
            dim_hidden = [128, 256, 512, 1024]
            kernel_size = 5
            stride = 2
            pad = 2
            shape_after_conv = 4

            init = chainer.initializers.HeNormal()
            self.conv1 = cl.Convolution2D(dim_in, dim_hidden[0], kernel_size, stride, pad, initialW=init)
            self.conv2 = cl.Convolution2D(dim_hidden[0] * 2, dim_hidden[1], kernel_size, stride, pad, initialW=init)
            self.conv3 = cl.Convolution2D(dim_hidden[1], dim_hidden[2], kernel_size, stride, pad, initialW=init)
            self.conv4 = cl.Convolution2D(dim_hidden[2], dim_hidden[3], kernel_size, stride, pad, initialW=init)
            self.linear_rm = cl.Linear(9, dim_hidden[0], initialW=init)
            self.linear_out = cl.Linear(dim_hidden[3] * shape_after_conv * shape_after_conv, 1, initialW=init)

    def __call__(self, x, rm):
        batch_size = x.shape[0]
        if x.shape[2] != 64:
            x = cf.resize_images(x, (64, 64))

        hi = cf.leaky_relu(self.conv1(x))  # [64 -> 32]
        hm = cf.leaky_relu(self.linear_rm(cf.reshape(rm, (batch_size, -1))))
        hm = cf.broadcast_to(hm[:, :, None, None], hi.shape)
        h = cf.concat((hi, hm), axis=1)

        h = cf.leaky_relu(self.conv2(h))  # [32 -> 16]
        h = cf.leaky_relu(self.conv3(h))  # [16 -> 8]
        h = cf.leaky_relu(self.conv4(h))  # [8 -> 4]
        h = cf.reshape(h, (batch_size, -1))
        y = self.linear_out(h)
        return y


class PascalPatchDiscriminator(chainer.Chain):
    def __init__(self, dim_in=4):
        super(PascalPatchDiscriminator, self).__init__()
        with self.init_scope():
            init = chainer.initializers.HeNormal()
            self.conv1 = cl.Convolution2D(dim_in, 64, 4, stride=2, pad=2, initialW=init)
            self.conv2 = cl.Convolution2D(64 * 2, 128, 4, stride=2, pad=2, initialW=init)
            self.conv3 = cl.Convolution2D(128, 256, 4, stride=2, pad=2, initialW=init)
            self.conv4 = cl.Convolution2D(256, 512, 4, stride=1, pad=2, initialW=init)
            self.conv5 = cl.Convolution2D(512, 1, 4, stride=1, pad=2, initialW=init)
            self.linear_rm = cl.Linear(9, 64, initialW=init)
            self.linear_labels = cl.Linear(3, 512, nobias=True, initialW=init)

    def __call__(self, x, rm, labels=None):
        hi = cf.leaky_relu(self.conv1(x))  # [224 -> 112]
        hm = cf.leaky_relu(self.linear_rm(cf.reshape(rm, (rm.shape[0], -1))))
        hm = cf.broadcast_to(hm[:, :, None, None], hi.shape)
        h = cf.concat((hi, hm), axis=1)

        h = cf.leaky_relu(self.conv2(h))  # [112 -> 56]
        h = cf.leaky_relu(self.conv3(h))  # [56 -> 28]
        h = cf.leaky_relu(self.conv4(h))  # [28 -> 14]
        h1 = self.conv5(h)

        if labels is not None:
            labels = to_one_hot_vector(labels, 3)
            h2 = self.linear_labels(labels)
            h2 = cf.broadcast_to(h2[:, :, None, None], h.shape)
            h2 = cf.sum(h2 * h, axis=1, keepdims=True)
            return h1[:, :, :-1, :-1] + h2
        else:
            return h1


class ShapeNetPatchDiscriminator(chainer.Chain):
    def __init__(self, dim_in=4):
        super(ShapeNetPatchDiscriminator, self).__init__()

        with self.init_scope():
            init = chainer.initializers.HeNormal()
            dims = [dim_in, 32, 64, 128, 256, 256, 1]
            Convolution2D = sn.sn_convolution_2d.SNConvolution2D
            Linear = sn.sn_linear.SNLinear
            self.conv1 = Convolution2D(dims[0], dims[1], 5, stride=2, pad=2, initialW=init, nobias=True)
            self.conv2 = Convolution2D(dims[1] * 2, dims[2], 5, stride=2, pad=2, initialW=init, nobias=True)
            self.conv3 = Convolution2D(dims[2], dims[3], 5, stride=2, pad=2, initialW=init, nobias=True)
            self.conv4 = Convolution2D(dims[3], dims[4], 5, stride=2, pad=2, initialW=init, nobias=True)
            self.conv5 = Convolution2D(dims[4], dims[5], 5, stride=2, pad=2, initialW=init, nobias=True)
            self.conv6 = Convolution2D(dims[5], dims[6], 5, stride=2, pad=2, initialW=init)
            self.linear_v = Linear(3, dims[1], initialW=init, nobias=True)
            self.linear_labels = Linear(13, dims[-2], nobias=True)

            self.conv1_bn = cl.BatchNormalization(dims[1], use_gamma=False)
            self.conv2_bn = cl.BatchNormalization(dims[2], use_gamma=False)
            self.conv3_bn = cl.BatchNormalization(dims[3], use_gamma=False)
            self.conv4_bn = cl.BatchNormalization(dims[4], use_gamma=False)
            self.conv5_bn = cl.BatchNormalization(dims[5], use_gamma=False)
            self.linear_v_bn = cl.BatchNormalization(dims[1], use_gamma=False)

    def __call__(self, x, v, labels=None):
        hi = cf.leaky_relu(self.conv1_bn(self.conv1(x)))  # [224 -> 112]
        hm = cf.leaky_relu(self.linear_v_bn(self.linear_v(v)))
        hm = cf.broadcast_to(hm[:, :, None, None], hi.shape)
        h = cf.concat((hi, hm), axis=1)

        h = cf.leaky_relu(self.conv2_bn(self.conv2(h)))  # [112 -> 56]
        h = cf.leaky_relu(self.conv3_bn(self.conv3(h)))  # [56 -> 28]
        h = cf.leaky_relu(self.conv4_bn(self.conv4(h)))  # [28 -> 14]
        h = cf.leaky_relu(self.conv5_bn(self.conv5(h)))  # [14 -> 7]
        h1 = self.conv6(h)  # [7 -> 4]

        if labels is not None:
            labels = to_one_hot_vector(labels, 13)
            h2 = self.linear_labels(labels)  # [bs, 256]
            h2 = cf.broadcast_to(h2[:, :, None, None], h.shape)  # [bs, 256, 7, 7]
            h2 = cf.sum(h2 * h, axis=1, keepdims=True)  # [bs, 1, 7, 7]
            h2 = h2[:, :, ::2, ::2]
            return h1 + h2
        else:
            return h1


class PascalPatchDiscriminator3(chainer.Chain):
    def __init__(self, dim_in=4, use_bn=True):
        super(PascalPatchDiscriminator3, self).__init__()

        with self.init_scope():
            init = chainer.initializers.HeNormal()
            dims = [dim_in, 32, 64, 128, 256, 256, 1]
            Convolution2D = sn.sn_convolution_2d.SNConvolution2D
            Linear = sn.sn_linear.SNLinear
            if use_bn:
                Normalization = cl.BatchNormalization
                no_bias = True
            else:
                import layers
                Normalization = layers.DummyLayer
                no_bias = False

            self.conv1 = Convolution2D(dims[0], dims[1], 5, stride=2, pad=2, initialW=init, nobias=no_bias)
            self.conv2 = Convolution2D(dims[1] * 2, dims[2], 5, stride=2, pad=2, initialW=init, nobias=no_bias)
            self.conv3 = Convolution2D(dims[2], dims[3], 5, stride=2, pad=2, initialW=init, nobias=no_bias)
            self.conv4 = Convolution2D(dims[3], dims[4], 5, stride=2, pad=2, initialW=init, nobias=no_bias)
            self.conv5 = Convolution2D(dims[4], dims[5], 5, stride=2, pad=2, initialW=init, nobias=no_bias)
            self.conv6 = Convolution2D(dims[5], dims[6], 5, stride=2, pad=2, initialW=init)
            self.linear_v = Linear(9, dims[1], initialW=init, nobias=True)
            self.linear_labels = Linear(3, dims[-2], nobias=True)

            if use_bn:
                self.conv1_bn = Normalization(dims[1], use_gamma=False)
                self.conv2_bn = Normalization(dims[2], use_gamma=False)
                self.conv3_bn = Normalization(dims[3], use_gamma=False)
                self.conv4_bn = Normalization(dims[4], use_gamma=False)
                self.conv5_bn = Normalization(dims[5], use_gamma=False)
                self.linear_v_bn = Normalization(dims[1], use_gamma=False)
            else:
                self.conv1_bn = layers.DummyLayer()
                self.conv2_bn = layers.DummyLayer()
                self.conv3_bn = layers.DummyLayer()
                self.conv4_bn = layers.DummyLayer()
                self.conv5_bn = layers.DummyLayer()
                self.linear_v_bn = layers.DummyLayer()

    def __call__(self, x, v, labels=None):
        hi = cf.leaky_relu(self.conv1_bn(self.conv1(x)))  # [224 -> 112]
        hm = cf.leaky_relu(self.linear_v_bn(self.linear_v(v)))
        hm = cf.broadcast_to(hm[:, :, None, None], hi.shape)
        h = cf.concat((hi, hm), axis=1)

        h = cf.leaky_relu(self.conv2_bn(self.conv2(h)))  # [112 -> 56]
        h = cf.leaky_relu(self.conv3_bn(self.conv3(h)))  # [56 -> 28]
        h = cf.leaky_relu(self.conv4_bn(self.conv4(h)))  # [28 -> 14]
        h = cf.leaky_relu(self.conv5_bn(self.conv5(h)))  # [14 -> 7]
        h1 = self.conv6(h)  # [7 -> 4]

        if labels is not None:
            labels = to_one_hot_vector(labels, 3)
            h2 = self.linear_labels(labels)  # [bs, 256]
            h2 = cf.broadcast_to(h2[:, :, None, None], h.shape)  # [bs, 256, 7, 7]
            h2 = cf.sum(h2 * h, axis=1, keepdims=True)  # [bs, 1, 7, 7]
            h2 = h2[:, :, ::2, ::2]
            return h1 + h2
        else:
            return h1


class PascalPatchDiscriminator2(chainer.Chain):
    def __init__(self, dim_in=4):
        super(PascalPatchDiscriminator2, self).__init__()
        with self.init_scope():
            init = chainer.initializers.HeNormal()
            Convolution2D = sn.sn_convolution_2d.SNConvolution2D
            Linear = sn.sn_linear.SNLinear
            self.conv1 = Convolution2D(dim_in, 64, 4, stride=2, pad=2, initialW=init)
            self.conv2 = Convolution2D(64 * 2, 128, 4, stride=2, pad=2, initialW=init)
            self.conv3 = Convolution2D(128, 256, 4, stride=2, pad=2, initialW=init)
            self.conv4 = Convolution2D(256, 512, 4, stride=2, pad=2, initialW=init)
            self.conv5 = Convolution2D(512, 1, 4, stride=1, pad=2, initialW=init)
            self.linear_rm = Linear(9, 64, initialW=init)

    def __call__(self, x, rm):
        hi = cf.leaky_relu(self.conv1(x))  # [224 -> 112]
        hm = cf.leaky_relu(self.linear_rm(cf.reshape(rm, (rm.shape[0], -1))))
        hm = cf.broadcast_to(hm[:, :, None, None], hi.shape)
        h = cf.concat((hi, hm), axis=1)

        h = cf.leaky_relu(self.conv2(h))  # [112 -> 56]
        h = cf.leaky_relu(self.conv3(h))  # [56 -> 28]
        h = cf.leaky_relu(self.conv4(h))  # [28 -> 14]
        h = self.conv5(h)
        return h


class PascalPatchDiscriminator4(chainer.Chain):
    def __init__(self, dim_in=4):
        super(PascalPatchDiscriminator4, self).__init__()

        with self.init_scope():
            init = chainer.initializers.HeNormal()
            dims = [dim_in, 32, 64, 128, 256, 256, 1]

            self.conv1 = cl.Convolution2D(dims[0], dims[1], 5, stride=2, pad=2, initialW=init)
            self.conv2 = cl.Convolution2D(dims[1] * 2, dims[2], 5, stride=2, pad=2, initialW=init)
            self.conv3 = cl.Convolution2D(dims[2], dims[3], 5, stride=2, pad=2, initialW=init)
            self.conv4 = cl.Convolution2D(dims[3], dims[4], 5, stride=2, pad=2, initialW=init)
            self.conv5 = cl.Convolution2D(dims[4], dims[5], 5, stride=2, pad=2, initialW=init)
            self.conv6 = cl.Convolution2D(dims[5], dims[6], 5, stride=2, pad=2, initialW=init)
            self.linear_v = cl.Linear(9, dims[1], initialW=init, nobias=True)
            self.linear_labels = cl.Linear(3, dims[-2], nobias=True)

    def __call__(self, x, v, labels=None):
        hi = cf.leaky_relu(self.conv1(x))  # [224 -> 112]
        hm = cf.leaky_relu(self.linear_v(v))
        hm = cf.broadcast_to(hm[:, :, None, None], hi.shape)
        h = cf.concat((hi, hm), axis=1)

        h = cf.leaky_relu(self.conv2(h))  # [112 -> 56]
        h = cf.leaky_relu(self.conv3(h))  # [56 -> 28]
        h = cf.leaky_relu(self.conv4(h))  # [28 -> 14]
        h = cf.leaky_relu(self.conv5(h))  # [14 -> 7]
        h1 = self.conv6(h)  # [7 -> 4]

        if labels is not None:
            labels = to_one_hot_vector(labels, 3)
            h2 = self.linear_labels(labels)  # [bs, 256]
            h2 = cf.broadcast_to(h2[:, :, None, None], h.shape)  # [bs, 256, 7, 7]
            h2 = cf.sum(h2 * h, axis=1, keepdims=True)  # [bs, 1, 7, 7]
            h2 = h2[:, :, ::2, ::2]
            return h1 + h2
        else:
            return h1


def get_discriminator(name, dim_in):
    if name == 'basic':
        return BasicDiscriminator(dim_in)
    elif name == 'pascal_basic':
        return PascalBasicDiscriminator(dim_in)
    elif name == 'shapenet_patch':
        return ShapeNetPatchDiscriminator(dim_in)
    elif name == 'pascal_patch':
        return PascalPatchDiscriminator(dim_in)
    elif name == 'pascal_patch2':
        return PascalPatchDiscriminator2(dim_in)
    elif name == 'pascal_patch3':
        return PascalPatchDiscriminator3(dim_in)
    elif name == 'pascal_patch4':
        return PascalPatchDiscriminator4(dim_in)
