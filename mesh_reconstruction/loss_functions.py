import chainer
import chainer.functions as cf
import neural_renderer

import perceptual_loss


def silhouette_loss(target, prediction, num_levels=5):
    batch_size = target.shape[0]
    loss_list = []
    t2 = target[:, None, :, :]
    p2 = prediction[:, None, :, :]
    for i in range(num_levels):
        if i != 0:
            t2 = cf.average_pooling_2d(t2, 2, 2)
            p2 = cf.average_pooling_2d(p2, 2, 2)
        t3 = cf.normalize(cf.reshape(t2, (batch_size, -1)))
        p3 = cf.normalize(cf.reshape(p2, (batch_size, -1)))
        loss_list.append(cf.sum(cf.square(t3 - p3)) / batch_size)
    loss = sum(loss_list)
    return loss


def silhouette_iou_loss(target, prediction):
    eps = 1e-5
    axes = tuple(range(target.ndim)[1:])
    intersection = cf.sum(target * prediction, axis=axes)
    union = cf.sum(target + prediction - target * prediction, axis=axes)
    iou = cf.sum(intersection / (union + eps)) / intersection.size
    return 1 - iou


def loss_textures(images1, images2, num_levels=5):
    assert not isinstance(images1, chainer.Variable)
    assert isinstance(images2, chainer.Variable)

    # [-1, 1] -> [0, 1]
    images1 = (images1 / 2 + 0.5)

    loss_list = []
    for i in range(num_levels):
        if i != 0:
            images1 = cf.average_pooling_2d(images1, 2, 2)
            images2 = cf.average_pooling_2d(images2, 2, 2)
        diff = cf.absolute(images1[:, :3] - images2[:, :3])
        loss = cf.mean(diff)
        loss_list.append(loss)
    loss = sum(loss_list)
    return loss


def perceptual_texture_loss(images1, images2, zero_mean=True):
    if zero_mean:
        # [-1, 1] -> [0, 1]
        images1 = (images1 / 2 + 0.5)
    loss = perceptual_loss.alex_net_loss(images1[:, :3], images2[:, :3])
    loss = cf.mean(loss)
    return loss


def adversarial_loss(data, loss_type=None):
    if loss_type == 'kl_real':
        return cf.mean(cf.softplus(-data))
    elif loss_type == 'kl_fake':
        return cf.mean(cf.softplus(data))
    elif loss_type == 'ls_d_real':
        return cf.mean(cf.square(data - 1))
    elif loss_type == 'ls_d_fake':
        return cf.mean(cf.square(data))
    elif loss_type == 'ls_g':
        return cf.mean(cf.square(data - 1))
    elif loss_type == 'hinge_d_real':
        return cf.mean(cf.relu(1. - data))
    elif loss_type == 'hinge_d_fake':
        return cf.mean(cf.relu(1. + data))
    elif loss_type == 'hinge_g':
        return -cf.mean(data)


def inflation_loss(vertices, faces, degrees, eps=1e-5):
    assert vertices.ndim == 3
    assert faces.ndim == 2

    v0 = vertices[:, faces[:, 0], :]
    v1 = vertices[:, faces[:, 1], :]
    v2 = vertices[:, faces[:, 2], :]
    batch_size, num_faces = v0.shape[:2]
    v0 = cf.reshape(v0, (batch_size * num_faces, 3))
    v1 = cf.reshape(v1, (batch_size * num_faces, 3))
    v2 = cf.reshape(v2, (batch_size * num_faces, 3))
    norms = neural_renderer.cross(v1 - v0, v2 - v0)  # [bs * nf, 3]
    norms = cf.normalize(norms)
    v0_t = (v0 + norms).data
    v1_t = (v1 + norms).data
    v2_t = (v2 + norms).data
    loss_v0 = cf.sum(cf.sqrt(cf.sum(cf.square(v0_t - v0), 1) + eps))
    loss_v1 = cf.sum(cf.sqrt(cf.sum(cf.square(v1_t - v1), 1) + eps))
    loss_v2 = cf.sum(cf.sqrt(cf.sum(cf.square(v2_t - v2), 1) + eps))
    loss = loss_v0 + loss_v1 + loss_v2
    loss /= batch_size
    return loss


def graph_laplacian_loss(vertices, laplacian, norm=True, eps=1e-5):
    # vertices: [bs, nv, 3]
    batch_size = vertices.shape[0]
    xp = chainer.cuda.get_array_module(laplacian)
    laplacian = laplacian / xp.diag(laplacian)
    laplacian = xp.tile(laplacian[None, :, :], (batch_size, 1, 1))
    vertices = cf.matmul(laplacian, vertices)  # [bs, nv, 3]
    diff = cf.sum(cf.square(vertices), axis=-1)
    if norm:
        diff = cf.sqrt(diff + eps)
    loss = cf.sum(diff) / batch_size
    return loss


def edge_length_loss(vertices, faces):
    assert vertices.ndim == 3
    assert faces.ndim == 2

    v0 = vertices[:, faces[:, 0], :]  # [bs, nf, 3]
    v1 = vertices[:, faces[:, 1], :]
    v2 = vertices[:, faces[:, 2], :]
    l01 = cf.mean(cf.square(v0 - v1))
    l12 = cf.mean(cf.square(v1 - v2))
    l20 = cf.mean(cf.square(v2 - v0))
    return l01 + l12 + l20
