import string

import chainer


def pad_zeros(x, size, axis, side='both'):
    xp = chainer.cuda.get_array_module(x)
    if axis == 1:
        pad = xp.zeros((x.shape[0], size, x.shape[2], x.shape[3]), 'float32')
    elif axis == 2:
        pad = xp.zeros((x.shape[0], x.shape[1], size, x.shape[3]), 'float32')
    elif axis == 3:
        pad = xp.zeros((x.shape[0], x.shape[1], x.shape[2], size), 'float32')
    if side == 'both':
        x = xp.concatenate((pad, x, pad), axis=axis)
    elif side == 'left':
        x = xp.concatenate((pad, x), axis=axis)
    elif side == 'right':
        x = xp.concatenate((x, pad), axis=axis)
    return x


def maximum(data_right, data_left, eps=1e-4):
    data3 = chainer.cuda.elementwise(
        'float32 data_right, float32 data_left',
        'float32 data_out',
        string.Template('''
            if (max(data_right, data_left) <= 0) {
                data_out = 0;
            } else if (abs(data_right - data_left) < ${eps}) {
                data_out = 0;
            } else if (data_right > data_left) {
                data_out = -data_right;
            } else {
                data_out = data_left;
            }
        ''').substitute(
            eps=eps,
        ),
        'function',
    )(data_right, data_left)
    return data3


class Differentiation(chainer.Function):
    def check_type_forward(self, in_types):
        # images: [bs, is, is, x]
        # coordinates: [bs, is, is, 2]
        chainer.utils.type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].ndim == 4,
            in_types[0].shape[1] == in_types[0].shape[2],
            in_types[1].dtype.kind == 'f',
            in_types[1].ndim == 4,
            in_types[1].shape[1] == in_types[1].shape[2],
            in_types[1].shape[3] == 2,
            in_types[0].shape[0] == in_types[1].shape[0],
            in_types[0].shape[1] == in_types[1].shape[1],
        )

    def forward_cpu(self, inputs):
        raise NotImplementedError

    def backward_cpu(self, inputs, grad_outputs):
        raise NotImplementedError

    def forward_gpu(self, inputs):
        images, coordinates = inputs
        return images,

    def backward_gpu(self, inputs, gradients):
        xp = chainer.cuda.get_array_module(inputs[0])
        images, coordinates = inputs
        grad_output = gradients[0]
        batch_size, image_size, _, num_channels = images.shape
        step = 2. / image_size

        grad_images = gradients[0].copy()

        grad_y_r = -((images[:, :-1, :] - images[:, 1:, :]) * grad_output[:, 1:, :]).sum(-1) / step
        grad_y_r = pad_zeros(grad_y_r[:, :, :, None], 1, 1, 'right') + pad_zeros(grad_y_r[:, :, :, None], 1, 1, 'left')
        grad_y_l = -((images[:, 1:, :] - images[:, :-1, :]) * grad_output[:, :-1, :]).sum(-1) / step
        grad_y_l = pad_zeros(grad_y_l[:, :, :, None], 1, 1, 'left') + pad_zeros(grad_y_l[:, :, :, None], 1, 1, 'right')
        grad_y = maximum(grad_y_r, grad_y_l)

        grad_x_r = -((images[:, :, :-1] - images[:, :, 1:]) * grad_output[:, :, 1:]).sum(-1) / step
        grad_x_r = pad_zeros(grad_x_r[:, :, :, None], 1, 2, 'right') + pad_zeros(grad_x_r[:, :, :, None], 1, 2, 'left')
        grad_x_l = -((images[:, :, 1:] - images[:, :, :-1]) * grad_output[:, :, :-1]).sum(-1) / step
        grad_x_l = pad_zeros(grad_x_l[:, :, :, None], 1, 2, 'left') + pad_zeros(grad_x_l[:, :, :, None], 1, 2, 'right')
        grad_x = maximum(grad_x_r, grad_x_l)

        grad_loss_xy = xp.concatenate((grad_x, grad_y), axis=-1)

        return grad_images, grad_loss_xy


def differentiation(images, coordinates):
    return Differentiation()(images, coordinates)
