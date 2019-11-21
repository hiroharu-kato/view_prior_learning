import functools
import string

import chainer
import chainer.functions as cf
import cupy as cp

from . import differentiation
from . import lights as light_lib

########################################################################################################################
# Parameters

# camera
DEFAULT_NEAR = 0.1
DEFAULT_FAR = 100

# rendering
DEFAULT_IMAGE_SIZE = 256
DEFAULT_ANTI_ALIASING = True
DEFAULT_DRAW_BACKSIDE = True

# others
DEFAULT_EPS = 1e-5


########################################################################################################################
# Utility functions


class ToMap(chainer.Function):
    """
    Test code:
    import chainer.gradient_check
    data_in = cp.random.randn(*(16, 128, 3, 5)).astype('float32')
    indices = cp.random.randint(-1, 128, size=(16, 8, 8)).astype('int32')
    grad_out = cp.random.randn(16, 8, 8, 3, 5).astype('float32')
    data_out = ToMap()(data_in, indices)
    for i1 in range(16):
        for i2 in range(8):
            for i3 in range(8):
                i4 = indices[i1, i2, i3]
                if i4 < 0:
                    chainer.testing.assert_allclose(data_out[i1, i2, i3].data, data_out[i1, i2, i3].data * 0)
                else:
                    chainer.testing.assert_allclose(data_out[i1, i2, i3].data, data_in[i1, i4])
    chainer.gradient_check.check_backward(ToMap(), (data_in, indices), grad_out, no_grads=(False, True), rtol=1e-2, atol=1e-03)
    """

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[1].dtype.kind == 'i',
            in_types[0].shape[0] == in_types[1].shape[0],
        )

    def forward_gpu(self, inputs):
        # data_in: [bs, nf, ...]
        # indices: [bs, is, is]
        # data_out: [bs, is, is, ..]
        data_in, indices = list(map(cp.ascontiguousarray, inputs))
        data_out = cp.ascontiguousarray(cp.zeros(tuple(list(indices.shape[:3]) + list(data_in.shape[2:])), 'float32'))
        chainer.cuda.elementwise(
            'raw float32 data_in, int32 index, raw float32 data_out',
            '',
            string.Template('''
                if (index < 0) return;

                int bn = i / (${image_size} * ${image_size});
                int pos_from = bn * ${num_features} * ${dim} + index * ${dim};
                int pos_to = i * ${dim};
                float* p1 = (float*)&data_in[pos_from];
                float* p2 = (float*)&data_out[pos_to];
                for (int j = 0; j < ${dim}; j++) *p2++ = *p1++;
            ''').substitute(
                image_size=indices.shape[1],
                num_features=data_in.shape[1],
                dim=functools.reduce(lambda x, y: x * y, data_in.shape[2:]),
            ),
            'function',
        )(
            data_in, indices, data_out,
        )
        return data_out,

    def backward_gpu(self, inputs, gradients):
        # data_in: [bs, nf, ...]
        # indices: [bs, is, is]
        # data_out: [bs, is, is, ..]
        data_in_shape = inputs[0].shape
        indices = cp.ascontiguousarray(inputs[1])
        grad_out = cp.ascontiguousarray(gradients[0])
        grad_in = cp.ascontiguousarray(cp.zeros(data_in_shape, 'float32'))
        chainer.cuda.elementwise(
            'raw float32 grad_in, int32 index, raw float32 grad_out',
            '',
            string.Template('''
                if (index < 0) return;

                int bn = i / (${image_size} * ${image_size});
                int pos_from = bn * ${num_features} * ${dim} + index * ${dim};
                int pos_to = i * ${dim};
                float* p1 = (float*)&grad_in[pos_from];
                float* p2 = (float*)&grad_out[pos_to];
                for (int j = 0; j < ${dim}; j++) atomicAdd(p1++, *p2++);
            ''').substitute(
                image_size=indices.shape[1],
                num_features=data_in_shape[1],
                dim=functools.reduce(lambda x, y: x * y, data_in_shape[2:]),
            ),
            'function',
        )(
            grad_in, indices, grad_out,
        )
        return grad_in, None


def to_map(data, indices):
    return ToMap()(data, indices)


class MaskForeground(chainer.Function):
    """
    Test code:
    import chainer.gradient_check
    data_in = cp.random.randn(*(16, 3, 3, 5)).astype('float32')
    masks = cp.random.randint(0, 2, size=(16, 3, 3)).astype('int32')
    grad_out = cp.random.randn(16, 3, 3, 5).astype('float32')
    data_out = mask_foreground(data_in, masks)
    for i1 in range(16):
        for i2 in range(3):
            for i3 in range(3):
                i4 = masks[i1, i2, i3]
                if 0 <= i4:
                    chainer.testing.assert_allclose(data_out[i1, i2, i3].data, data_in[i1, i2, i3])
                else:
                    chainer.testing.assert_allclose(data_out[i1, i2, i3].data, data_out[i1, i2, i3].data * 0)
    chainer.gradient_check.check_backward(
        mask_foreground, (data_in, masks), grad_out, no_grads=(False, True), rtol=1e-2, atol=1e-03)
    """

    def forward_gpu(self, inputs):
        data_in, face_index_map = list(map(cp.ascontiguousarray, inputs))
        if data_in.ndim == 3:
            dim = 1
        else:
            dim = functools.reduce(lambda x, y: x * y, data_in.shape[3:])
        data_out = cp.ascontiguousarray(cp.zeros(data_in.shape, 'float32'))
        chainer.cuda.elementwise(
            'int32 face_index, raw float32 data_in, raw float32 data_out',
            '',
            string.Template('''
                if (0 <= face_index) {
                    float* p1 = (float*)&data_in[i * ${dim}];
                    float* p2 = (float*)&data_out[i * ${dim}];
                    for (int j = 0; j < ${dim}; j++) *p2++ = *p1++;
                }
            ''').substitute(
                dim=dim,
            ),
            'function',
        )(face_index_map, data_in, data_out)
        return data_out,

    def backward_gpu(self, inputs, gradients):
        face_index_map = cp.ascontiguousarray(inputs[1])
        grad_out = cp.ascontiguousarray(gradients[0])
        grad_in = cp.ascontiguousarray(cp.zeros(grad_out.shape, 'float32'))
        if grad_in.ndim == 3:
            dim = 1
        else:
            dim = functools.reduce(lambda x, y: x * y, grad_in.shape[3:])
        chainer.cuda.elementwise(
            'int32 face_index, raw float32 data_in, raw float32 data_out',
            '',
            string.Template('''
                if (0 <= face_index) {
                    float* p1 = (float*)&data_in[i * ${dim}];
                    float* p2 = (float*)&data_out[i * ${dim}];
                    for (int j = 0; j < ${dim}; j++) *p1++ = *p2++;
                }
            ''').substitute(
                dim=dim,
            ),
            'function',
        )(face_index_map, grad_in, grad_out)
        return grad_in, None


def mask_foreground(data, face_index_map):
    return MaskForeground()(data, face_index_map)


########################################################################################################################
# Core functions


class FaceIndexMap(chainer.Function):
    def __init__(self, num_faces, image_size, near, far, draw_backside):
        self.num_faces = num_faces
        self.image_size = image_size
        self.near = near
        self.far = far
        self.draw_backside = draw_backside

    def forward_gpu(self, inputs):
        return self.forward_gpu_safe(inputs)
        # return self.forward_gpu_unsafe(inputs)

    def forward_gpu_safe(self, inputs):
        xp = chainer.cuda.get_array_module(inputs[0])
        faces = inputs[0]
        batch_size, num_faces = faces.shape[:2]
        faces = xp.ascontiguousarray(faces)

        loop = xp.arange(batch_size * self.image_size * self.image_size).astype('int32')
        face_index_map = chainer.cuda.elementwise(
            'int32 _, raw float32 faces',
            'int32 face_index',
            string.Template('''
                const int is = ${image_size};
                const int nf = ${num_faces};
                const int bn = i / (is * is);
                const int pn = i % (is * is);
                const int yi = pn / is;
                const int xi = pn % is;
                const float yp = (2. * yi + 1 - is) / is;
                const float xp = (2. * xi + 1 - is) / is;

                float* face = (float*)&faces[bn * nf * 9];
                float depth_min = ${far};
                int face_index_min = -1;
                for (int fn = 0; fn < nf; fn++) {
                    /* go to next face */
                    const float x0 = *face++;
                    const float y0 = *face++;
                    const float z0 = *face++;
                    const float x1 = *face++;
                    const float y1 = *face++;
                    const float z1 = *face++;
                    const float x2 = *face++;
                    const float y2 = *face++;
                    const float z2 = *face++;

                    if (xp < x0 && xp < x1 && xp < x2) continue;
                    if (x0 < xp && x1 < xp && x2 < xp) continue;
                    if (yp < y0 && yp < y1 && yp < y2) continue;
                    if (y0 < yp && y1 < yp && y2 < yp) continue;

                    /* return if backside */
                    if (!${draw_backside}) {
                        if ((y2 - y0) * (x1 - x0) > (y1 - y0) * (x2 - x0)) continue;
                    }

                    /* check in or out */
                    float c1 = (yp - y0) * (x1 - x0) - (y1 - y0) * (xp - x0);
                    float c2 = (yp - y1) * (x2 - x1) - (y2 - y1) * (xp - x1);
                    if (c1 * c2 < 0) continue;
                    float c3 = (yp - y2) * (x0 - x2) - (y0 - y2) * (xp - x2);
                    if (c2 * c3 < 0) continue;

                    float det = x2 * (y0 - y1) + x0 * (y1 - y2) + x1 * (y2 - y0);
                    if (abs(det) < 0.00000001) continue;

                    /* */
                    if (depth_min < z0 && depth_min < z1 && depth_min < z2) continue;

                    /* compute w */
                    float w[3];
                    w[0] = yp * (x2 - x1) + xp * (y1 - y2) + (x1 * y2 - x2 * y1);
                    w[1] = yp * (x0 - x2) + xp * (y2 - y0) + (x2 * y0 - x0 * y2);
                    w[2] = yp * (x1 - x0) + xp * (y0 - y1) + (x0 * y1 - x1 * y0);
                    const float w_sum = w[0] + w[1] + w[2];
                    w[0] /= w_sum;
                    w[1] /= w_sum;
                    w[2] /= w_sum;

                    /* compute 1 / zp = sum(w / z) */
                    const float zp = 1. / (w[0] / z0 + w[1] / z1 + w[2] / z2);
                    if (zp <= ${near} || ${far} <= zp) continue;

                    /* check z-buffer */
                    if (zp <= depth_min) {
                        depth_min = zp;
                        face_index_min = fn;
                    }
                }

                /* set to global memory */
                face_index = face_index_min;
            ''').substitute(
                num_faces=self.num_faces,
                image_size=self.image_size,
                near=self.near,
                far=self.far,
                draw_backside=int(self.draw_backside),
                eps=1e-8,
            ),
            'function',
        )(loop, faces)
        face_index_map = face_index_map.reshape((batch_size, self.image_size, self.image_size))

        return face_index_map,

    def forward_gpu_unsafe(self, inputs):
        xp = chainer.cuda.get_array_module(inputs[0])
        faces = inputs[0]
        batch_size, num_faces = faces.shape[:2]
        faces = xp.ascontiguousarray(faces)

        loop = xp.arange(batch_size * self.num_faces).astype('int32')
        face_index_map = cp.zeros((batch_size, self.image_size, self.image_size), 'int32') - 1
        depth_map = cp.zeros((batch_size, self.image_size, self.image_size), 'float32') * self.far + 1
        lock = cp.zeros((batch_size, self.image_size, self.image_size), 'int32')
        chainer.cuda.elementwise(
            'int32 _, raw float32 faces, raw int32 face_index_map, raw float32 depth_map',
            'raw int32 lock',
            string.Template('''
                const int is = ${image_size};
                const int fn = i % ${num_faces};
                const int bn = i / ${num_faces};

                float* face = (float*)&faces[i * 9];
                const float x0 = *face++;
                const float y0 = *face++;
                const float z0 = *face++;
                const float x1 = *face++;
                const float y1 = *face++;
                const float z1 = *face++;
                const float x2 = *face++;
                const float y2 = *face++;
                const float z2 = *face++;
                const float xp_min = min(x0, min(x1, x2));
                const float xp_max = max(x0, max(x1, x2));
                const float yp_min = min(y0, min(y1, y2));
                const float yp_max = max(y0, max(y1, y2));
                const int xi_min = ceil((xp_min * is + is - 1) / 2.);
                const int xi_max = floor((xp_max * is + is - 1) / 2.);
                const int yi_min = ceil((yp_min * is + is - 1) / 2.);
                const int yi_max = floor((yp_max * is + is - 1) / 2.);
                for (int xi = xi_min; xi <= xi_max; xi++) {
                    for (int yi = yi_min; yi <= yi_max; yi++) {
                        const int pi = bn * is * is + yi * is + xi;
                        const float yp = (2. * yi + 1 - is) / is;
                        const float xp = (2. * xi + 1 - is) / is;

                        if (xp < x0 && xp < x1 && xp < x2) continue;
                        if (x0 < xp && x1 < xp && x2 < xp) continue;
                        if (yp < y0 && yp < y1 && yp < y2) continue;
                        if (y0 < yp && y1 < yp && y2 < yp) continue;

                        /* return if backside */
                        if (!${draw_backside}) {
                            if ((y2 - y0) * (x1 - x0) > (y1 - y0) * (x2 - x0)) continue;
                        }

                        /* check in or out */
                        float c1 = (yp - y0) * (x1 - x0) - (y1 - y0) * (xp - x0);
                        float c2 = (yp - y1) * (x2 - x1) - (y2 - y1) * (xp - x1);
                        if (c1 * c2 < 0) continue;
                        float c3 = (yp - y2) * (x0 - x2) - (y0 - y2) * (xp - x2);
                        if (c2 * c3 < 0) continue;

                        float det = x2 * (y0 - y1) + x0 * (y1 - y2) + x1 * (y2 - y0);
                        if (abs(det) < 0.00000001) continue;

                        /* compute w */
                        float w[3];
                        w[0] = yp * (x2 - x1) + xp * (y1 - y2) + (x1 * y2 - x2 * y1);
                        w[1] = yp * (x0 - x2) + xp * (y2 - y0) + (x2 * y0 - x0 * y2);
                        w[2] = yp * (x1 - x0) + xp * (y0 - y1) + (x0 * y1 - x1 * y0);
                        const float w_sum = w[0] + w[1] + w[2];
                        w[0] /= w_sum;
                        w[1] /= w_sum;
                        w[2] /= w_sum;

                        /* compute 1 / zp = sum(w / z) */
                        const float zp = 1. / (w[0] / z0 + w[1] / z1 + w[2] / z2);
                        if (zp <= ${near} || ${far} <= zp) continue;

                        unsigned int* l = (unsigned int*)&lock[pi];
                        while (atomicCAS(l, 0 , 1) != 0);
                        /* check z-buffer */
                        /*float depth_min = depth_map[pi];
                        if (zp <= depth_min) {
                            atomicExch((float*)&depth_map[pi], zp);
                            atomicExch((int*)&face_index_map[pi], fn);
                        } */
                        atomicExch(l, 0) ;
                    }
                }

            ''').substitute(
                num_faces=self.num_faces,
                image_size=self.image_size,
                near=self.near,
                far=self.far,
                draw_backside=int(self.draw_backside),
                eps=1e-8,
            ),
            'function',
        )(loop, faces, face_index_map, depth_map, lock)
        face_index_map = face_index_map.reshape((batch_size, self.image_size, self.image_size))

        return face_index_map,


def compute_face_index_map(faces, image_size, near, far, draw_backside):
    batch_size, num_faces = faces.shape[:2]
    face_index_map = FaceIndexMap(num_faces, image_size, near, far, draw_backside)(faces).data
    return face_index_map


def compute_weight_map(faces, face_index_map):
    xp = chainer.cuda.get_array_module(faces)
    batch_size, num_faces = faces.shape[:2]
    image_size = face_index_map.shape[1]

    weight_map = xp.zeros((batch_size * image_size * image_size, 3), 'float32')
    if isinstance(faces, chainer.Variable):
        faces = faces.data
    faces = xp.ascontiguousarray(faces)
    face_index_map = xp.ascontiguousarray(face_index_map)
    weight_map = xp.ascontiguousarray(weight_map)
    chainer.cuda.elementwise(
        'raw float32 faces, int32 face_index_map, raw float32 weight_map',
        '',
        string.Template('''
            const int fi = face_index_map;
            if (fi < 0) continue;

            const int is = ${image_size};
            const int nf = ${num_faces};
            const int bn = i / (is * is);
            const int pn = i % (is * is);
            const int yi = pn / is;
            const int xi = pn % is;
            const float yp = (2. * yi + 1 - is) / is;
            const float xp = (2. * xi + 1 - is) / is;

            float* face = (float*)&faces[(bn * nf + fi) * 9];
            float x0 = *face++;
            float y0 = *face++;
            float z0 = *face++;
            float x1 = *face++;
            float y1 = *face++;
            float z1 = *face++;
            float x2 = *face++;
            float y2 = *face++;
            float z2 = *face++;

            /* compute w */
            float w[3];
            w[0] = yp * (x2 - x1) + xp * (y1 - y2) + (x1 * y2 - x2 * y1);
            w[1] = yp * (x0 - x2) + xp * (y2 - y0) + (x2 * y0 - x0 * y2);
            w[2] = yp * (x1 - x0) + xp * (y0 - y1) + (x0 * y1 - x1 * y0);
            float w_sum = w[0] + w[1] + w[2];
            if (w_sum < 0) {
                w[0] *= -1;
                w[1] *= -1;
                w[2] *= -1;
            }
            w[0] = max(w[0], 0.);
            w[1] = max(w[1], 0.);
            w[2] = max(w[2], 0.);
            w_sum = w[0] + w[1] + w[2];
            float* wm = (float*)&weight_map[i * 3];
            for (int j = 0; j < 3; j++) {
                w[j] /= w_sum;
                w[j] = max(min(w[j], 1.), 0.);
                wm[j] = w[j];
            }
        ''').substitute(
            num_faces=num_faces,
            image_size=image_size,
        ),
        'function',
    )(faces, face_index_map.flatten(), weight_map)
    weight_map = weight_map.reshape((batch_size, image_size, image_size, 3))

    return weight_map


def compute_depth_map(faces, face_index_map, weight_map):
    # faces: [bs, nf, 3, 3]
    # face_index_map: [bs, is, is]
    # weight_map: [bs, is, is, 3]

    faces_z_map = to_map(faces[:, :, :, -1:], face_index_map)[:, :, :, :, 0]  # [bs, is, is, 3]
    depth_map = 1. / cf.sum(weight_map / faces_z_map, axis=-1)
    depth_map = mask_foreground(depth_map, face_index_map)
    return depth_map


def compute_silhouettes(face_index_map):
    return (0 <= face_index_map).astype('float32')


def compute_coordinate_map(faces, face_index_map, weight_map):
    # faces: [bs, nf, 3, 3]
    # face_index_map: [bs, is, is]
    # weight_map: [bs, is, is, 3]
    faces_map = to_map(faces[:, :, :, :2], face_index_map)  # [bs, is, is, 3, 3]
    coordinate_map = cf.sum(faces_map * weight_map[:, :, :, :, None], axis=-2)
    return coordinate_map


def sample_textures(faces, faces_textures, textures, face_index_map, weight_map, eps):
    # faces: [bs, nf, 3, 3]
    # faces_textures: [bs, nf, 3, 2]
    # textures: [bs, 3, is, is]
    # face_index_map: [bs, is, is]
    # weight_map: [bs, is, is, 3]
    xp = chainer.cuda.get_array_module(faces)
    batch_size, num_faces = faces.shape[:2]
    texture_height, texture_width = textures.shape[2:]
    if isinstance(faces, chainer.Variable):
        faces = faces.data
    if isinstance(faces_textures, chainer.Variable):
        faces_textures = faces_textures.data
    if isinstance(face_index_map, chainer.Variable):
        face_index_map = face_index_map.data
    if isinstance(weight_map, chainer.Variable):
        weight_map = weight_map.data

    textures = cf.transpose(textures, (0, 2, 3, 1))  # [bs, h, w, 3]
    textures = cf.reshape(textures, (batch_size, texture_height * texture_width, 3))  # [bs, h * w, 3]
    faces_z_map = to_map(faces[:, :, :, 2], face_index_map).data  # [bs, is, is, 3]
    vertices_textures_map = to_map(faces_textures, face_index_map).data  # [bs, is, is, 3, 2]
    depth_map = 1. / (weight_map / faces_z_map).sum(-1)  # [bs, is, is]

    # -> [bs, is, is, 2]
    vertices_textures_map_original = vertices_textures_map.copy()
    vertices_textures_map = (
            weight_map[:, :, :, :, None] * vertices_textures_map / faces_z_map[:, :, :, :, None]).sum(-2)
    vertices_textures_map = vertices_textures_map * depth_map[:, :, :, None]  # [bs, is, is, 2]
    vertices_textures_map = xp.maximum(vertices_textures_map, vertices_textures_map_original.min(-2))
    vertices_textures_map = xp.minimum(vertices_textures_map, vertices_textures_map_original.max(-2) - eps)
    vertices_textures_map = mask_foreground(vertices_textures_map, face_index_map).data

    x_f = vertices_textures_map[:, :, :, 0]
    y_f = vertices_textures_map[:, :, :, 1]
    x_f_f = xp.floor(x_f)
    y_f_f = xp.floor(y_f)
    x_c_f = x_f_f + 1
    y_c_f = y_f_f + 1
    x_f_i = x_f_f.astype('int32')
    y_f_i = y_f_f.astype('int32')
    x_c_i = x_c_f.astype('int32')
    y_c_i = y_c_f.astype('int32')

    #
    vtm1 = (y_f_i * texture_width + x_f_i)  # [bs, is, is]
    vtm2 = (y_f_i * texture_width + x_c_i)  # [bs, is, is]
    vtm3 = (y_c_i * texture_width + x_f_i)  # [bs, is, is]
    vtm4 = (y_c_i * texture_width + x_c_i)  # [bs, is, is]
    w1 = (y_c_f - y_f) * (x_c_f - x_f)  # [bs * is * is]
    w2 = (y_c_f - y_f) * (x_f - x_f_f)  # [bs, is, is]
    w3 = (y_f - y_f_f) * (x_c_f - x_f)  # [bs, is, is]
    w4 = (y_f - y_f_f) * (x_f - x_f_f)  # [bs, is, is]
    images = (
            w1[:, :, :, None] * to_map(textures, vtm1) +
            w2[:, :, :, None] * to_map(textures, vtm2) +
            w3[:, :, :, None] * to_map(textures, vtm3) +
            w4[:, :, :, None] * to_map(textures, vtm4))

    # mask foreground
    images = mask_foreground(images, face_index_map)

    return images


def blend_backgrounds(face_index_map, rgb_map, backgrounds):
    foreground_map = (0 <= face_index_map).astype('float32')[:, :, :, None]  # [bs, is, is, 1]
    rgb_map = foreground_map * rgb_map + (1 - foreground_map) * backgrounds[:, ::-1, ::-1]
    return rgb_map


def compute_normal_map(vertices, face_indices, faces, face_index_map, weight_map, smooth=True):
    # faces: [bs, nf, 3, 3]
    # face_index_map: [bs, is, is]
    # weight_map: [bs, is, is, 3]

    from . import cross
    v01 = faces[:, :, 1, :] - faces[:, :, 0, :]
    v12 = faces[:, :, 2, :] - faces[:, :, 1, :]
    v01 = cf.reshape(v01, (-1, 3))
    v12 = cf.reshape(v12, (-1, 3))
    n = cross(v01, v12)
    n = cf.reshape(n, (faces.shape[0], faces.shape[1], 3))  # [bs, nf, 3]
    m = cp.zeros((face_indices.shape[0], vertices.shape[1]), 'float32')  # [nf, nv]
    m[cp.arange(m.shape[0]), face_indices[:, 0]] = 1
    m[cp.arange(m.shape[0]), face_indices[:, 1]] = 1
    m[cp.arange(m.shape[0]), face_indices[:, 2]] = 1
    n = n.transpose((0, 2, 1))  # [bs, 3, nf]
    n = cf.reshape(n, (-1, n.shape[-1]))  # [bs * 3, nf]
    n = cf.matmul(n, m)
    n = cf.reshape(n, (faces.shape[0], 3, vertices.shape[1]))  # [bs, 3, nv]
    n = n.transpose((0, 2, 1))
    n = cf.normalize(n, axis=2)  # [bs, nv, 3]
    n = n[:, face_indices]  # [bs, nv, 3, 3]

    normal_map = to_map(n, face_index_map)  # [bs, is, is, 3, 3]
    if smooth:
        normal_map = cf.sum(weight_map[:, :, :, :, None] * normal_map, axis=-2)
    else:
        normal_map = cf.mean(normal_map, axis=-2)
    return normal_map


########################################################################################################################
# Interfaces
def rasterize_all(
        vertices,
        faces,
        vertices_textures=None,
        faces_textures=None,
        textures=None,
        background_color=None,
        backgrounds=None,
        lights=None,
        image_size=DEFAULT_IMAGE_SIZE,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        draw_backside=DEFAULT_DRAW_BACKSIDE,
        draw_rgb=True,
        draw_silhouettes=True,
        draw_depth=True,
):
    # vertices: [batch_size, num_vertices, 3]
    # faces: [num_faces, 3]
    # vertices_textures: [batch_size, num_vertices_textures, 2]
    # faces_textures: [num_faces, 3]
    # textures: [batch_size, 3, height, width]
    assert vertices.ndim == 3
    assert vertices.shape[2] == 3
    assert faces.ndim == 2
    assert faces.shape[1] == 3
    if draw_rgb:
        assert vertices_textures.ndim == 3
        assert vertices_textures.shape[2] == 2
        assert faces_textures.ndim == 2
        assert faces_textures.shape[1] == 3
        assert textures.ndim == 4
        assert textures.shape[1] == 3
    if background_color is not None:
        xp = chainer.cuda.get_array_module(vertices)
        if anti_aliasing:
            backgrounds = xp.zeros((vertices.shape[0], 3, image_size * 2, image_size * 2), 'float32')
        else:
            backgrounds = xp.zeros((vertices.shape[0], 3, image_size, image_size), 'float32')
        backgrounds = backgrounds * xp.array(background_color)[None, :, None, None]
    elif backgrounds is not None:
        assert backgrounds.ndim == 4
        assert backgrounds.shape[0] == vertices.shape[0]
        assert backgrounds.shape[1] == 3
        if anti_aliasing:
            assert backgrounds.shape[2] == image_size * 2
            assert backgrounds.shape[3] == image_size * 2
        else:
            assert backgrounds.shape[2] == image_size
            assert backgrounds.shape[3] == image_size

    if anti_aliasing:
        image_size *= 2

    # -> [batch_size, num_faces, 3, 3]
    face_indices = faces.copy()
    faces = vertices[:, faces]

    # -> [batch_size, num_faces, 3, 3]
    face_index_map = compute_face_index_map(faces, image_size, near, far, draw_backside)

    # -> [batch_size, image_size, image_size, 3]
    weight_map = compute_weight_map(faces, face_index_map)

    # -> [batch_size, 1, image_size, image_size]
    if draw_silhouettes or backgrounds is not None:
        silhouettes = compute_silhouettes(face_index_map)[:, :, :, None]

    if draw_rgb:
        # -> [batch_size, num_faces, 3, 3]
        faces_textures = vertices_textures[:, faces_textures]

        # -> [batch_size, image_size, image_size, 3]
        rgb_map = sample_textures(faces, faces_textures, textures, face_index_map, weight_map, eps)

        if lights is not None:
            normal_map = compute_normal_map(vertices, face_indices, faces, face_index_map, weight_map)
            color_weight_map = chainer.Variable(cp.zeros(normal_map.shape, 'float32'))
            for light in lights:
                if isinstance(light, light_lib.AmbientLight):
                    color_weight_map += cf.broadcast_to(light.color[:, None, None, :], color_weight_map.shape)
                if isinstance(light, light_lib.DirectionalLight):
                    # [bs, is, is]
                    intensity = cf.sum(
                        cf.broadcast_to(-light.direction[:, None, None, :], normal_map.shape) * normal_map, -1)
                    if light.backside:
                        intensity = cf.absolute(intensity)
                    else:
                        intensity = cf.relu(intensity)
                    intensity = cf.broadcast_to(intensity[:, :, :, None], color_weight_map.shape)
                    color = cf.broadcast_to(light.color[:, None, None, :], color_weight_map.shape)
                    color_weight_map += intensity * color
                if isinstance(light, light_lib.SpecularLight):
                    # [bs, is, is]
                    direction_eye = cp.array([0, 0, 1], 'float32')
                    intensity = cf.sum(-direction_eye[None, None, None, :] * normal_map, -1)
                    if light.backside:
                        intensity = cf.absolute(intensity)
                    else:
                        intensity = cf.relu(intensity)
                    intensity **= light.alpha[:, None, None]
                    intensity = cf.broadcast_to(intensity[:, :, :, None], color_weight_map.shape)
                    color = cf.broadcast_to(light.color[:, None, None, :], color_weight_map.shape)
                    color_weight_map += intensity * color
            rgb_map *= color_weight_map

        # blend backgrounds
        if backgrounds is not None:
            backgrounds = backgrounds.transpose((0, 2, 3, 1))
            rgb_map = blend_backgrounds(face_index_map, rgb_map, backgrounds)

    # -> [batch_size, 1, image_size, image_size]
    if draw_depth:
        depth_map = compute_depth_map(faces, face_index_map, weight_map)[:, :, :, None]

    # merge
    if draw_rgb and draw_silhouettes and draw_depth:
        images = cf.concat((rgb_map, silhouettes, depth_map), axis=-1)
    elif draw_rgb and draw_silhouettes and not draw_depth:
        images = cf.concat((rgb_map, silhouettes), axis=-1)
    elif draw_rgb and not draw_silhouettes and draw_depth:
        images = cf.concat((rgb_map, depth_map), axis=-1)
    elif draw_rgb and not draw_silhouettes and not draw_depth:
        images = rgb_map
    elif not draw_rgb and draw_silhouettes and draw_depth:
        images = cf.concat((silhouettes, depth_map), axis=-1)
    elif not draw_rgb and draw_silhouettes and not draw_depth:
        images = silhouettes
    elif not draw_rgb and not draw_silhouettes and draw_depth:
        images = depth_map
    elif not draw_rgb and not draw_silhouettes and not draw_depth:
        raise Exception

    # -> [batch_size, image_size, image_size, 2]
    coordinate_map = compute_coordinate_map(faces, face_index_map, weight_map)

    # -> [batch_size, 3, image_size, image_size]
    images = differentiation.differentiation(images, coordinate_map)
    images = images[:, ::-1, ::-1, :].transpose((0, 3, 1, 2))

    # down sampling
    if anti_aliasing:
        # average pooling. faster than cf.average_pooling_2d(images, 2, 2)
        images = (
                images[:, :, 0::2, 0::2] +
                images[:, :, 1::2, 0::2] +
                images[:, :, 0::2, 1::2] +
                images[:, :, 1::2, 1::2])
        images /= 4.
    return images


def rasterize_silhouettes(
        vertices,
        faces,
        background_color=None,
        backgrounds=None,
        image_size=DEFAULT_IMAGE_SIZE,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        draw_backside=DEFAULT_DRAW_BACKSIDE,
):
    images = rasterize_all(
        vertices=vertices,
        faces=faces,
        background_color=background_color,
        backgrounds=backgrounds,
        image_size=image_size,
        near=near,
        far=far,
        eps=eps,
        anti_aliasing=anti_aliasing,
        draw_backside=draw_backside,
        draw_rgb=False,
        draw_silhouettes=True,
        draw_depth=False,
    )
    return images[:, 0]


def rasterize_rgba(
        vertices,
        faces,
        vertices_textures,
        faces_textures,
        textures,
        background_color=None,
        backgrounds=None,
        lights=None,
        image_size=DEFAULT_IMAGE_SIZE,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        draw_backside=DEFAULT_DRAW_BACKSIDE,
):
    images = rasterize_all(
        vertices=vertices,
        faces=faces,
        vertices_textures=vertices_textures,
        faces_textures=faces_textures,
        textures=textures,
        background_color=background_color,
        backgrounds=backgrounds,
        lights=lights,
        image_size=image_size,
        near=near,
        far=far,
        eps=eps,
        anti_aliasing=anti_aliasing,
        draw_backside=draw_backside,
        draw_rgb=True,
        draw_silhouettes=True,
        draw_depth=False,
    )
    return images


def rasterize_rgb(
        vertices,
        faces,
        vertices_textures,
        faces_textures,
        textures,
        background_color=None,
        backgrounds=None,
        lights=None,
        image_size=DEFAULT_IMAGE_SIZE,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        draw_backside=DEFAULT_DRAW_BACKSIDE,
):
    images = rasterize_all(
        vertices=vertices,
        faces=faces,
        vertices_textures=vertices_textures,
        faces_textures=faces_textures,
        textures=textures,
        background_color=background_color,
        backgrounds=backgrounds,
        lights=lights,
        image_size=image_size,
        near=near,
        far=far,
        eps=eps,
        anti_aliasing=anti_aliasing,
        draw_backside=draw_backside,
        draw_rgb=True,
        draw_silhouettes=False,
        draw_depth=False,
    )
    return images


def rasterize(
        vertices,
        faces,
        vertices_textures,
        faces_textures,
        textures,
        background_color=None,
        backgrounds=None,
        lights=None,
        image_size=DEFAULT_IMAGE_SIZE,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        draw_backside=DEFAULT_DRAW_BACKSIDE,
):
    images = rasterize_all(
        vertices=vertices,
        faces=faces,
        vertices_textures=vertices_textures,
        faces_textures=faces_textures,
        textures=textures,
        background_color=background_color,
        backgrounds=backgrounds,
        lights=lights,
        image_size=image_size,
        near=near,
        far=far,
        eps=eps,
        anti_aliasing=anti_aliasing,
        draw_backside=draw_backside,
        draw_rgb=True,
        draw_silhouettes=False,
        draw_depth=False,
    )
    return images


def rasterize_depth(
        vertices,
        faces,
        backgrounds=None,
        background_color=None,
        image_size=DEFAULT_IMAGE_SIZE,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        draw_backside=DEFAULT_DRAW_BACKSIDE,
):
    images = rasterize_all(
        vertices=vertices,
        faces=faces,
        background_color=background_color,
        backgrounds=backgrounds,
        image_size=image_size,
        near=near,
        far=far,
        eps=eps,
        anti_aliasing=anti_aliasing,
        draw_backside=draw_backside,
        draw_rgb=False,
        draw_silhouettes=False,
        draw_depth=True,
    )
    return images[:, 0]
