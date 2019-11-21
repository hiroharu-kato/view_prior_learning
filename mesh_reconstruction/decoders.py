import chainer
import chainer.cuda
import chainer.functions as cf
import chainer.links as cl
import neural_renderer

import layers


def get_graph_laplacian(faces, num_vertices):
    xp = chainer.cuda.get_array_module(faces)
    laplacian = xp.zeros((num_vertices, num_vertices), 'float32')
    for face in faces:
        i0, i1, i2 = face
        laplacian[i0, i1] = -1
        laplacian[i1, i0] = -1
        laplacian[i1, i2] = -1
        laplacian[i2, i1] = -1
        laplacian[i2, i0] = -1
        laplacian[i0, i2] = -1
    laplacian -= xp.identity(num_vertices) * laplacian.sum(0)
    return laplacian


def up_sample(x):
    h, w = x.shape[2:]
    return cf.unpooling_2d(x, 2, outsize=(h * 2, w * 2))


class BasicShapeDecoder(chainer.Chain):
    def __init__(self, dim_in=512, scaling=1., filename_obj='./data/obj/sphere_642.obj'):
        super(BasicShapeDecoder, self).__init__()

        with self.init_scope():
            self.vertices_base, self.faces = neural_renderer.load_obj(filename_obj)
            self.num_vertices = self.vertices_base.shape[0]
            self.num_faces = self.faces.shape[0]
            self.obj_scale = 0.5
            self.object_size = 1.0
            self.scaling = scaling

            dim_hidden = [4096, 4096]
            init = chainer.initializers.HeNormal()
            self.linear1 = cl.Linear(dim_in, dim_hidden[0], initialW=init)
            self.linear2 = cl.Linear(dim_hidden[0], dim_hidden[1], initialW=init)
            self.linear_bias = cl.Linear(dim_hidden[1], self.num_vertices * 3, initialW=init)

            self.laplacian = get_graph_laplacian(self.faces, self.num_vertices)

    def to_gpu(self, device=None):
        super(BasicShapeDecoder, self).to_gpu(device)
        self.vertices_base = chainer.cuda.to_gpu(self.vertices_base, device)
        self.faces = chainer.cuda.to_gpu(self.faces, device)
        self.laplacian = chainer.cuda.to_gpu(self.laplacian, device)

    def __call__(self, x):
        h = cf.relu(self.linear1(x))
        h = cf.relu(self.linear2(h))
        bias = self.linear_bias(h) * self.scaling
        bias = cf.reshape(bias, (-1, self.num_vertices, 3))

        base = self.vertices_base * self.obj_scale
        base = self.xp.broadcast_to(base[None, :, :], bias.shape)

        vertices = self.object_size * cf.tanh(base + bias) * 0.99

        return vertices, self.faces
    

class BasicSymmetricShapeDecoder(chainer.Chain):
    def __init__(self, dim_in=512, scaling=1., filename_obj='./data/obj/sphere_642.obj'):
        super(BasicSymmetricShapeDecoder, self).__init__()

        with self.init_scope():
            self.vertices_base, self.faces = neural_renderer.load_obj(filename_obj)
            self.num_vertices = self.vertices_base.shape[0]
            self.num_faces = self.faces.shape[0]
            self.obj_scale = 0.5
            self.object_size = 1.0
            self.scaling = scaling

            self.vertices_base, self.vertices_matrix = self.compute_vertices_matrix()  # [642 * 3, 337 * 3]
            dim_out = self.vertices_matrix.shape[1]

            dim_hidden = [4096, 4096]
            init = chainer.initializers.HeNormal()
            self.linear1 = cl.Linear(dim_in, dim_hidden[0], initialW=init)
            self.linear2 = cl.Linear(dim_hidden[0], dim_hidden[1], initialW=init)
            self.linear_bias = cl.Linear(dim_hidden[1], dim_out, initialW=init)

            self.laplacian = get_graph_laplacian(self.faces, self.num_vertices)
            self.degrees = self.xp.histogram(self.faces, self.xp.arange(self.num_vertices + 1))[0].astype('int32')

    def compute_vertices_matrix(self):
        # create matrix to convert predicted 337 vertices to symmetric 642 vertices
        vertices_unique = []
        for v in self.vertices_base:
            if self.xp.absolute(v[2]) < 1e-3:
                vertices_unique.append((v[0], v[1], 0))
            elif 0 < v[2]:
                vertices_unique.append((v[0], v[1], v[2]))
        vertices_unique = self.xp.array(vertices_unique)

        vertices_matrix = self.xp.zeros((self.vertices_base.shape[0], 3, vertices_unique.shape[0], 3), 'float32')
        for i, v in enumerate(self.vertices_base):
            v2 = self.xp.array((v[0], v[1], self.xp.absolute(v[2])))
            dists = ((vertices_unique - v2[None, :]) ** 2).sum(-1)
            j = dists.argmin()
            vertices_matrix[i, 0, j, 0] = 1
            vertices_matrix[i, 1, j, 1] = 1
            if self.xp.absolute(v[2]) < 1e-3:
                vertices_matrix[i, 2, j, 2] = 0
            elif 0 < v[2]:
                vertices_matrix[i, 2, j, 2] = 1
            else:
                vertices_matrix[i, 2, j, 2] = -1
        vertices_matrix = vertices_matrix.reshape((vertices_matrix.shape[0] * 3, -1))
        return vertices_unique, vertices_matrix

    def to_gpu(self, device=None):
        super(BasicSymmetricShapeDecoder, self).to_gpu(device)
        self.vertices_base = chainer.cuda.to_gpu(self.vertices_base, device)
        self.faces = chainer.cuda.to_gpu(self.faces, device)
        self.laplacian = chainer.cuda.to_gpu(self.laplacian, device)
        self.vertices_matrix = chainer.cuda.to_gpu(self.vertices_matrix, device)
        self.degrees = chainer.cuda.to_gpu(self.degrees, device)

    def __call__(self, x):
        batch_size = x.shape[0]

        # predict 337 vertices [bs, 337, 3]
        h = cf.relu(self.linear1(x))
        h = cf.relu(self.linear2(h))
        vertices = self.linear_bias(h) * self.scaling
        vertices = vertices.reshape((batch_size, -1, 3))

        # add base sphere and normalize
        base = self.vertices_base * self.obj_scale
        base = self.xp.broadcast_to(base[None, :, :], vertices.shape)
        vertices = vertices + base
        vertices = self.object_size * cf.tanh(vertices) * 0.99

        # z <- abs(z)
        xy = vertices[:, :, :2]
        z = cf.absolute(vertices[:, :, 2:3])
        vertices = cf.concat((xy, z), axis=2)

        # assign to 642 vertices
        # bias: [bs, 337, 3]
        # vertices_matrix: [642 * 3, 337 * 3]
        vertices = cf.reshape(vertices, (batch_size, -1))
        vertices_matrix = self.xp.tile(self.vertices_matrix[None, :, :], (batch_size, 1, 1))
        vertices = cf.matmul(vertices_matrix, vertices[:, :, None])
        vertices = cf.reshape(vertices, (batch_size, -1, 3))

        return vertices, self.faces


class ConvolutionShapeDecoder(chainer.Chain):
    def __init__(
            self, dim_in=512, scaling=1.0, use_bn=True, use_up_sampling=False, symmetric=False):
        super(ConvolutionShapeDecoder, self).__init__()

        self.grid_size = 16
        self.obj_scale = 0.5
        self.tanh_scale = 1.2
        self.scaling = scaling
        self.use_up_sampling = use_up_sampling
        self.symmetric = symmetric

        # create base shape & faces and transforming matrix
        self.vertices_base = None
        self.vertices_matrix = None
        self.num_vertices = None
        self.symmetric_matrix = None
        self.faces = None
        self.degrees = None
        self.init_vertices_base()
        self.init_faces()
        self.laplacian = get_graph_laplacian(self.faces, self.num_vertices)
        self.normalize_vertices_base()

        # init NN layers
        with self.init_scope():
            dim_h = [512, 256, 128, 64, 3]
            init = chainer.initializers.HeNormal()
            layer_list = {}
            if use_bn:
                Normalization = cl.BatchNormalization
                no_bias = True
            else:
                Normalization = layers.DummyLayer
                no_bias = False

            for i in range(6):
                layer_list['linear_p%d_1' % i] = cl.Linear(dim_in, dim_h[0] * 4, initialW=init, nobias=no_bias)
                if not use_up_sampling:
                    layer_list['conv_p%d_1' % i] = (
                        cl.Deconvolution2D(dim_h[0], dim_h[1], 3, 2, 1, outsize=(4, 4), initialW=init, nobias=no_bias))
                    layer_list['conv_p%d_2' % i] = (
                        cl.Deconvolution2D(dim_h[1], dim_h[2], 3, 2, 1, outsize=(8, 8), initialW=init, nobias=no_bias))
                    layer_list['conv_p%d_3' % i] = (
                        cl.Deconvolution2D(dim_h[2], dim_h[3], 3, 2, 1, outsize=(16, 16), initialW=init,
                                           nobias=no_bias))
                else:
                    layer_list['conv_p%d_1' % i] = (
                        cl.Convolution2D(dim_h[0], dim_h[1], 3, pad=1, initialW=init, nobias=no_bias))
                    layer_list['conv_p%d_2' % i] = (
                        cl.Convolution2D(dim_h[1], dim_h[2], 3, pad=1, initialW=init, nobias=no_bias))
                    layer_list['conv_p%d_3' % i] = (
                        cl.Convolution2D(dim_h[2], dim_h[3], 3, pad=1, initialW=init, nobias=no_bias))
                layer_list['conv_p%d_4' % i] = (cl.Convolution2D(dim_h[3], dim_h[4], 3, 1, 1, initialW=init))
                layer_list['linear_p%d_1_bn' % i] = Normalization(dim_h[0])
                layer_list['conv_p%d_1_bn' % i] = Normalization(dim_h[1])
                layer_list['conv_p%d_2_bn' % i] = Normalization(dim_h[2])
                layer_list['conv_p%d_3_bn' % i] = Normalization(dim_h[3])
            for k, v in layer_list.items():
                setattr(self, k, v)

            self.vertices_base = chainer.Parameter(self.vertices_base)

    def init_vertices_base(self):
        grid_size = self.grid_size

        # vertices:
        # vertices_matrix: transform 6x(16^2) coordinates from convolution layers into (16^2-14^2) coordinates
        vertices = []
        vertices_matrix = self.xp.zeros((6 * grid_size ** 2, (grid_size ** 3 - (grid_size - 2) ** 3)), 'float32')
        for z in range(grid_size):
            for y in range(grid_size):
                for x in range(grid_size):
                    if (z in [0, grid_size - 1]) or (y in [0, grid_size - 1]) or (x in [0, grid_size - 1]):
                        vertices.append((x, y, z))
                        vn = len(vertices) - 1
                        if z == 0:
                            vn2 = (grid_size ** 2) * 0 + y * grid_size + x
                            vertices_matrix[vn2, vn] = 1
                        if z == grid_size - 1:
                            vn2 = (grid_size ** 2) * 1 + y * grid_size + x
                            vertices_matrix[vn2, vn] = 1
                        if y == 0:
                            vn2 = (grid_size ** 2) * 2 + x * grid_size + z
                            vertices_matrix[vn2, vn] = 1
                        if y == grid_size - 1:
                            vn2 = (grid_size ** 2) * 3 + x * grid_size + z
                            vertices_matrix[vn2, vn] = 1
                        if x == 0:
                            vn2 = (grid_size ** 2) * 4 + z * grid_size + y
                            vertices_matrix[vn2, vn] = 1
                        if x == grid_size - 1:
                            vn2 = (grid_size ** 2) * 5 + z * grid_size + y
                            vertices_matrix[vn2, vn] = 1

        vertices_matrix /= vertices_matrix.sum(0)[None, :]
        vertices = self.xp.array(vertices)

        self.vertices_base = vertices
        self.vertices_matrix = vertices_matrix
        self.num_vertices = vertices.shape[0]

        symmetric_matrix = self.xp.zeros((vertices.shape[0], vertices.shape[0]), 'float32')
        z_sign = self.xp.zeros(vertices.shape[0], 'float32')
        v1 = vertices.copy()
        v2 = vertices.copy()
        v2[:, -1] = 15 - v2[:, -1]
        d = ((v1[:, None, :] - v2[None, :, :]) ** 2).sum(-1)
        for i in range(vertices.shape[0]):
            symmetric_matrix[i, i] = 0.5
            symmetric_matrix[self.xp.nonzero((d[i] == 0))[0][0], i] = 0.5
            if v1[i, -1] < grid_size / 2.:
                z_sign[i] = -1
            else:
                z_sign[i] = 1
        self.symmetric_matrix = symmetric_matrix
        self.z_sign = z_sign

    def get_nearest_vertex(self, x):
        x = self.xp.array(x)
        return ((self.vertices_base - x[None, :]) ** 2).sum(-1).argmin()

    def init_faces(self):
        faces = []
        for axis in range(3):
            for d3 in [0, self.grid_size]:
                for d1 in range(self.grid_size - 1):
                    for d2 in range(self.grid_size - 1):
                        if axis == 0:
                            v1 = [d1, d2, d3]
                            v2 = [d1 + 1, d2, d3]
                            v3 = [d1, d2 + 1, d3]
                            v4 = [d1 + 1, d2 + 1, d3]
                        elif axis == 1:
                            v1 = [d3, d1, d2]
                            v2 = [d3, d1 + 1, d2]
                            v3 = [d3, d1, d2 + 1]
                            v4 = [d3, d1 + 1, d2 + 1]
                        elif axis == 2:
                            v1 = [d2, d3, d1]
                            v2 = [d2, d3, d1 + 1]
                            v3 = [d2 + 1, d3, d1]
                            v4 = [d2 + 1, d3, d1 + 1]
                        vn1 = self.get_nearest_vertex(v1)
                        vn2 = self.get_nearest_vertex(v2)
                        vn3 = self.get_nearest_vertex(v3)
                        vn4 = self.get_nearest_vertex(v4)
                        faces.append((vn1, vn2, vn4))
                        faces.append((vn1, vn4, vn3))
        self.faces = self.xp.array(faces, 'int32')
        self.degrees = self.xp.histogram(self.faces, self.xp.arange(self.num_vertices + 1))[0].astype('int32')

    def normalize_vertices_base(self):
        vertices_base = (self.vertices_base.astype('float32') / (self.grid_size - 1)) * 2 - 1  # -> [-1, 1]
        norm = ((vertices_base ** 2).sum(-1) ** 0.5)
        vertices_base /= norm[:, None]  # -> sphere [-1, 1]
        vertices_base *= self.obj_scale  # -> [-obj_scale, obj_scale]
        vertices_base = self.xp.arctanh(vertices_base / self.tanh_scale)
        self.vertices_base = vertices_base.astype('float32')

    def to_gpu(self, device=None):
        super(ConvolutionShapeDecoder, self).to_gpu(device)
        self.vertices_matrix = chainer.cuda.to_gpu(self.vertices_matrix, device)
        self.faces = chainer.cuda.to_gpu(self.faces, device)
        self.degrees = chainer.cuda.to_gpu(self.degrees, device)
        self.laplacian = chainer.cuda.to_gpu(self.laplacian, device)
        self.symmetric_matrix = chainer.cuda.to_gpu(self.symmetric_matrix, device)
        self.z_sign = chainer.cuda.to_gpu(self.z_sign, device)

    def __call__(self, x):
        batch_size = x.shape[0]
        hs = []
        for i in range(6):
            linear1 = getattr(self, 'linear_p%d_1' % i)
            conv1 = getattr(self, 'conv_p%d_1' % i)
            conv2 = getattr(self, 'conv_p%d_2' % i)
            conv3 = getattr(self, 'conv_p%d_3' % i)
            conv4 = getattr(self, 'conv_p%d_4' % i)
            linear1_bn = getattr(self, 'linear_p%d_1_bn' % i)
            conv1_bn = getattr(self, 'conv_p%d_1_bn' % i)
            conv2_bn = getattr(self, 'conv_p%d_2_bn' % i)
            conv3_bn = getattr(self, 'conv_p%d_3_bn' % i)

            h = linear1(x)
            h = h.reshape((h.shape[0], -1, 2, 2))
            h = cf.relu(linear1_bn(h))
            if not self.use_up_sampling:
                h = cf.relu(conv1_bn(conv1(h)))
                h = cf.relu(conv2_bn(conv2(h)))
                h = cf.relu(conv3_bn(conv3(h)))
            else:
                h = cf.relu(conv1_bn(conv1(up_sample(h))))
                h = cf.relu(conv2_bn(conv2(up_sample(h))))
                h = cf.relu(conv3_bn(conv3(up_sample(h))))
            h = conv4(h)  # [bs, 3, 16, 16]
            h = cf.transpose(h, (0, 2, 3, 1))  # [bs, 16, 16, 3]
            h = cf.reshape(h, (batch_size, -1, 3))  # [bs, 16 ** 2, 3]
            hs.append(h)

        h = cf.concat(hs, axis=1)  # [bs, 6 * 16 ** 2, 3]
        vm = self.xp.tile(self.vertices_matrix[None, :, :], (batch_size, 1, 1))
        bias = cf.matmul(vm, h, transa=True)
        bias *= self.scaling
        base = self.vertices_base
        base = cf.broadcast_to(base[None, :, :], bias.shape)
        vertices = base + bias
        if self.symmetric:
            xy = vertices[:, :, :2]  # [bs, nv, 2]
            z = cf.absolute(vertices[:, :, 2:3])  # [bs, nv, 1]
            vertices = cf.concat((xy, z), axis=2)

            vertices = cf.transpose(cf.tensordot(vertices, self.symmetric_matrix, axes=(1, 0)), (0, 2, 1))

            xy = vertices[:, :, :2]  # [bs, nv, 2]
            z = vertices[:, :, 2:3]  # [bs, nv, 1]
            z = z * self.z_sign[None, :, None]
            vertices = cf.concat((xy, z), axis=2)

        vertices = cf.tanh(vertices) * self.tanh_scale

        return vertices, self.faces


class FCShapeDecoder(chainer.Chain):
    def __init__(
            self, dim_in=512, scaling=1.0, use_bn=False, fc_connection=False, symmetric=False):
        super(FCShapeDecoder, self).__init__()

        self.grid_size = 16
        self.obj_scale = 0.5
        self.tanh_scale = 1.2
        self.scaling = scaling
        self.fc_connection = fc_connection
        self.symmetric = symmetric

        # create base shape & faces and transforming matrix
        self.vertices_base = None
        self.vertices_matrix = None
        self.num_vertices = None
        self.symmetric_matrix = None
        self.faces = None
        self.degrees = None
        self.init_vertices_base()
        self.init_faces()
        self.laplacian = get_graph_laplacian(self.faces, self.num_vertices)
        self.normalize_vertices_base()

        # init NN layers
        with self.init_scope():
            init = chainer.initializers.HeNormal()
            if use_bn:
                Normalization = cl.BatchNormalization
                no_bias = True
            else:
                Normalization = layers.DummyLayer
                no_bias = False

            self.linear1 = cl.Linear(dim_in, 4096, initialW=init, nobias=no_bias)
            self.linear2 = cl.Linear(4096, 4096, initialW=init, nobias=no_bias)
            self.linear_bias = cl.Linear(4096, self.num_vertices * 3, initialW=init)
            self.linear1_bn = Normalization(4096)
            self.linear2_bn = Normalization(4096)
            self.vertices_base = chainer.Parameter(self.vertices_base)

    def init_vertices_base(self):
        grid_size = self.grid_size

        # vertices:
        # vertices_matrix: transform 6x(16^2) coordinates from convolution layers into (16^2-14^2) coordinates
        vertices = []
        for z in range(grid_size):
            for y in range(grid_size):
                for x in range(grid_size):
                    if (z in [0, grid_size - 1]) or (y in [0, grid_size - 1]) or (x in [0, grid_size - 1]):
                        vertices.append((x, y, z))

        vertices = self.xp.array(vertices)

        self.vertices_base = vertices
        self.num_vertices = vertices.shape[0]

        symmetric_matrix = self.xp.zeros((vertices.shape[0], vertices.shape[0]), 'float32')
        z_sign = self.xp.zeros(vertices.shape[0], 'float32')
        v1 = vertices.copy()
        v2 = vertices.copy()
        v2[:, -1] = 15 - v2[:, -1]
        d = ((v1[:, None, :] - v2[None, :, :]) ** 2).sum(-1)
        for i in range(vertices.shape[0]):
            symmetric_matrix[i, i] = 0.5
            symmetric_matrix[self.xp.nonzero((d[i] == 0))[0][0], i] = 0.5
            if v1[i, -1] < grid_size / 2.:
                z_sign[i] = -1
            else:
                z_sign[i] = 1
        self.symmetric_matrix = symmetric_matrix
        self.z_sign = z_sign

    def get_nearest_vertex(self, x):
        x = self.xp.array(x)
        return ((self.vertices_base - x[None, :]) ** 2).sum(-1).argmin()

    def init_faces(self):
        faces = []
        for axis in range(3):
            for d3 in [0, self.grid_size]:
                for d1 in range(self.grid_size - 1):
                    for d2 in range(self.grid_size - 1):
                        if axis == 0:
                            v1 = [d1, d2, d3]
                            v2 = [d1 + 1, d2, d3]
                            v3 = [d1, d2 + 1, d3]
                            v4 = [d1 + 1, d2 + 1, d3]
                        elif axis == 1:
                            v1 = [d3, d1, d2]
                            v2 = [d3, d1 + 1, d2]
                            v3 = [d3, d1, d2 + 1]
                            v4 = [d3, d1 + 1, d2 + 1]
                        elif axis == 2:
                            v1 = [d2, d3, d1]
                            v2 = [d2, d3, d1 + 1]
                            v3 = [d2 + 1, d3, d1]
                            v4 = [d2 + 1, d3, d1 + 1]
                        vn1 = self.get_nearest_vertex(v1)
                        vn2 = self.get_nearest_vertex(v2)
                        vn3 = self.get_nearest_vertex(v3)
                        vn4 = self.get_nearest_vertex(v4)
                        faces.append((vn1, vn2, vn4))
                        faces.append((vn1, vn4, vn3))
        self.faces = self.xp.array(faces, 'int32')
        self.degrees = self.xp.histogram(self.faces, self.xp.arange(self.num_vertices + 1))[0].astype('int32')

    def normalize_vertices_base(self):
        vertices_base = (self.vertices_base.astype('float32') / (self.grid_size - 1)) * 2 - 1  # -> [-1, 1]
        norm = ((vertices_base ** 2).sum(-1) ** 0.5)
        vertices_base /= norm[:, None]  # -> sphere [-1, 1]
        vertices_base *= self.obj_scale  # -> [-obj_scale, obj_scale]
        vertices_base = self.xp.arctanh(vertices_base / self.tanh_scale)
        self.vertices_base = vertices_base.astype('float32')

    def to_gpu(self, device=None):
        super(FCShapeDecoder, self).to_gpu(device)
        self.faces = chainer.cuda.to_gpu(self.faces, device)
        self.degrees = chainer.cuda.to_gpu(self.degrees, device)
        self.laplacian = chainer.cuda.to_gpu(self.laplacian, device)
        self.symmetric_matrix = chainer.cuda.to_gpu(self.symmetric_matrix, device)
        self.z_sign = chainer.cuda.to_gpu(self.z_sign, device)

    def __call__(self, x):
        h = cf.relu(self.linear1_bn(self.linear1(x)))
        h = cf.relu(self.linear2_bn(self.linear2(h)))
        bias = cf.reshape(self.linear_bias(h), (-1, self.num_vertices, 3))
        bias *= self.scaling
        base = self.vertices_base
        base = cf.broadcast_to(base[None, :, :], bias.shape)
        vertices = base + bias
        if self.symmetric:
            xy = vertices[:, :, :2]  # [bs, nv, 2]
            z = cf.absolute(vertices[:, :, 2:3])  # [bs, nv, 1]
            vertices = cf.concat((xy, z), axis=2)

            vertices = cf.transpose(cf.tensordot(vertices, self.symmetric_matrix, axes=(1, 0)), (0, 2, 1))

            xy = vertices[:, :, :2]  # [bs, nv, 2]
            z = vertices[:, :, 2:3]  # [bs, nv, 1]
            z = z * self.z_sign[None, :, None]
            vertices = cf.concat((xy, z), axis=2)

        vertices = cf.tanh(vertices) * self.tanh_scale

        return vertices, self.faces


class ResNetShapeDecoder(ConvolutionShapeDecoder):
    def __init__(self, dim_in=512, scaling=1.0):
        super(ConvolutionShapeDecoder, self).__init__()

        self.grid_size = 16
        self.obj_scale = 0.5
        self.scaling = scaling

        # create base shape & faces and transforming matrix
        self.vertices_base = None
        self.vertices_matrix = None
        self.num_vertices = None
        self.faces = None
        self.init_vertices_base()
        self.init_faces()
        self.laplacian = get_graph_laplacian(self.faces, self.num_vertices)
        self.normalize_vertices_base()

        # init NN layers
        with self.init_scope():
            dh = [512, 256, 128, 64, 3]
            init = chainer.initializers.HeNormal()
            layers = {}
            for i in range(6):
                layers['linear_p%d_in' % i] = cl.Linear(dim_in, dh[0] * 4, initialW=init, nobias=True)

                layers['conv_p%d_1_1_1' % i] = (
                    cl.Deconvolution2D(dh[0], dh[1], 3, 2, 1, outsize=(4, 4), initialW=init, nobias=True))
                layers['conv_p%d_1_1_2' % i] = (cl.Convolution2D(dh[1], dh[1], 3, 1, 1, initialW=init, nobias=True))
                layers['conv_p%d_1_1_3' % i] = (
                    cl.Deconvolution2D(dh[0], dh[1], 1, 2, 0, outsize=(4, 4), initialW=init, nobias=True))
                layers['conv_p%d_1_2_1' % i] = (cl.Convolution2D(dh[1], dh[1], 3, 1, 1, initialW=init, nobias=True))
                layers['conv_p%d_1_2_2' % i] = (cl.Convolution2D(dh[1], dh[1], 3, 1, 1, initialW=init, nobias=True))

                layers['conv_p%d_2_1_1' % i] = (
                    cl.Deconvolution2D(dh[1], dh[2], 3, 2, 1, outsize=(8, 8), initialW=init, nobias=True))
                layers['conv_p%d_2_1_2' % i] = (cl.Convolution2D(dh[2], dh[2], 3, 1, 1, initialW=init, nobias=True))
                layers['conv_p%d_2_1_3' % i] = (
                    cl.Deconvolution2D(dh[1], dh[2], 1, 2, 0, outsize=(8, 8), initialW=init, nobias=True))
                layers['conv_p%d_2_2_1' % i] = (cl.Convolution2D(dh[2], dh[2], 3, 1, 1, initialW=init, nobias=True))
                layers['conv_p%d_2_2_2' % i] = (cl.Convolution2D(dh[2], dh[2], 3, 1, 1, initialW=init, nobias=True))

                layers['conv_p%d_3_1_1' % i] = (
                    cl.Deconvolution2D(dh[2], dh[3], 3, 2, 1, outsize=(16, 16), initialW=init, nobias=True))
                layers['conv_p%d_3_1_2' % i] = (cl.Convolution2D(dh[3], dh[3], 3, 1, 1, initialW=init, nobias=True))
                layers['conv_p%d_3_1_3' % i] = (
                    cl.Deconvolution2D(dh[2], dh[3], 1, 2, 0, outsize=(16, 16), initialW=init, nobias=True))
                layers['conv_p%d_3_2_1' % i] = (cl.Convolution2D(dh[3], dh[3], 3, 1, 1, initialW=init, nobias=True))
                layers['conv_p%d_3_2_2' % i] = (cl.Convolution2D(dh[3], dh[3], 3, 1, 1, initialW=init, nobias=True))

                layers['linear_p%d_out' % i] = cl.Convolution2D(dh[3], dh[4], 1, 1, 0, initialW=init)

                layers['linear_p%d_in_bn' % i] = cl.BatchNormalization(dh[0])
                layers['conv_p%d_1_1_2_bn' % i] = cl.BatchNormalization(dh[1])
                layers['conv_p%d_1_2_1_bn' % i] = cl.BatchNormalization(dh[1])
                layers['conv_p%d_1_2_2_bn' % i] = cl.BatchNormalization(dh[1])
                layers['conv_p%d_2_1_1_bn' % i] = cl.BatchNormalization(dh[1])
                layers['conv_p%d_2_1_2_bn' % i] = cl.BatchNormalization(dh[2])
                layers['conv_p%d_2_2_1_bn' % i] = cl.BatchNormalization(dh[2])
                layers['conv_p%d_2_2_2_bn' % i] = cl.BatchNormalization(dh[2])
                layers['conv_p%d_3_1_1_bn' % i] = cl.BatchNormalization(dh[2])
                layers['conv_p%d_3_1_2_bn' % i] = cl.BatchNormalization(dh[3])
                layers['conv_p%d_3_2_1_bn' % i] = cl.BatchNormalization(dh[3])
                layers['conv_p%d_3_2_2_bn' % i] = cl.BatchNormalization(dh[3])
                layers['linear_p%d_out_bn' % i] = cl.BatchNormalization(dh[3])
            for k, v in layers.items():
                setattr(self, k, v)
            self.vertices_base = chainer.Parameter(self.vertices_base)

    def __call__(self, x):
        batch_size = x.shape[0]
        hs = []
        for i in range(6):
            linear_in = getattr(self, 'linear_p%d_in' % i)
            conv1_1_1 = getattr(self, 'conv_p%d_1_1_1' % i)
            conv1_1_2 = getattr(self, 'conv_p%d_1_1_2' % i)
            conv1_1_3 = getattr(self, 'conv_p%d_1_1_3' % i)
            conv1_2_1 = getattr(self, 'conv_p%d_1_2_1' % i)
            conv1_2_2 = getattr(self, 'conv_p%d_1_2_2' % i)
            conv2_1_1 = getattr(self, 'conv_p%d_2_1_1' % i)
            conv2_1_2 = getattr(self, 'conv_p%d_2_1_2' % i)
            conv2_1_3 = getattr(self, 'conv_p%d_2_1_3' % i)
            conv2_2_1 = getattr(self, 'conv_p%d_2_2_1' % i)
            conv2_2_2 = getattr(self, 'conv_p%d_2_2_2' % i)
            conv3_1_1 = getattr(self, 'conv_p%d_3_1_1' % i)
            conv3_1_2 = getattr(self, 'conv_p%d_3_1_2' % i)
            conv3_1_3 = getattr(self, 'conv_p%d_3_1_3' % i)
            conv3_2_1 = getattr(self, 'conv_p%d_3_2_1' % i)
            conv3_2_2 = getattr(self, 'conv_p%d_3_2_2' % i)
            linear_out = getattr(self, 'linear_p%d_out' % i)
            linear_in_bn = getattr(self, 'linear_p%d_in_bn' % i)
            conv1_1_2_bn = getattr(self, 'conv_p%d_1_1_2_bn' % i)
            conv1_2_1_bn = getattr(self, 'conv_p%d_1_2_1_bn' % i)
            conv1_2_2_bn = getattr(self, 'conv_p%d_1_2_2_bn' % i)
            conv2_1_1_bn = getattr(self, 'conv_p%d_2_1_1_bn' % i)
            conv2_1_2_bn = getattr(self, 'conv_p%d_2_1_2_bn' % i)
            conv2_2_1_bn = getattr(self, 'conv_p%d_2_2_1_bn' % i)
            conv2_2_2_bn = getattr(self, 'conv_p%d_2_2_2_bn' % i)
            conv3_1_1_bn = getattr(self, 'conv_p%d_3_1_1_bn' % i)
            conv3_1_2_bn = getattr(self, 'conv_p%d_3_1_2_bn' % i)
            conv3_2_1_bn = getattr(self, 'conv_p%d_3_2_1_bn' % i)
            conv3_2_2_bn = getattr(self, 'conv_p%d_3_2_2_bn' % i)
            linear_out_bn = getattr(self, 'linear_p%d_out_bn' % i)

            # [1 -> 2]
            h = linear_in(x)
            h = h.reshape((h.shape[0], -1, 2, 2))
            h = linear_in_bn(h)

            # [2 -> 4]
            h1 = conv1_1_1(cf.relu(h))
            h1 = conv1_1_2(cf.relu(conv1_1_2_bn(h1)))
            h2 = conv1_1_3(h)
            h = h1 + h2
            h1 = conv1_2_1(cf.relu(conv1_2_1_bn(h)))
            h1 = conv1_2_2(cf.relu(conv1_2_2_bn(h1)))
            h = h + h1

            # [4 -> 8]
            h1 = conv2_1_1(cf.relu(conv2_1_1_bn(h)))
            h1 = conv2_1_2(cf.relu(conv2_1_2_bn(h1)))
            h2 = conv2_1_3(h)
            h = h1 + h2
            h1 = conv2_2_1(cf.relu(conv2_2_1_bn(h)))
            h1 = conv2_2_2(cf.relu(conv2_2_2_bn(h1)))
            h = h + h1

            # [8 -> 16]
            h1 = conv3_1_1(cf.relu(conv3_1_1_bn(h)))
            h1 = conv3_1_2(cf.relu(conv3_1_2_bn(h1)))
            h2 = conv3_1_3(h)
            h = h1 + h2
            h1 = conv3_2_1(cf.relu(conv3_2_1_bn(h)))
            h1 = conv3_2_2(cf.relu(conv3_2_2_bn(h1)))
            h = h + h1

            h = linear_out(cf.relu(linear_out_bn(h)))  # [bs, 3, 16, 16]
            h = cf.transpose(h, (0, 2, 3, 1))  # [bs, 16, 16, 3]
            h = cf.reshape(h, (batch_size, -1, 3))  # [bs, 16 ** 2, 3]
            hs.append(h)

        h = cf.concat(hs, axis=1)  # [bs, 6 * 16 ** 2, 3]
        vm = self.xp.tile(self.vertices_matrix[None, :, :], (batch_size, 1, 1))
        bias = cf.matmul(vm, h, transa=True)
        bias *= self.scaling
        base = self.vertices_base * self.obj_scale
        base = cf.broadcast_to(base[None, :, :], bias.shape)
        vertices = cf.tanh(base + bias)

        return vertices, self.faces


class ConvolutionTextureDecoder(chainer.Chain):
    def __init__(self, dim_in=512, scaling=1.0, symmetric=False):
        super(ConvolutionTextureDecoder, self).__init__()

        self.grid_size = 16
        self.texture_size = 64
        self.scaling = scaling
        self.symmetric = symmetric

        self.vertices = None
        self.faces = None
        self.compute_vertices()

        with self.init_scope():
            dim_out = 3
            dh = [512, 256, 128, 64]
            init = chainer.initializers.HeNormal()

            layer_list = {}
            for i in range(6):
                layer_list['linear_p%d_1' % i] = cl.Linear(dim_in, dh[0] * 4 * 4, initialW=init, nobias=True)
                layer_list['conv_p%d_1' % i] = (
                    cl.Deconvolution2D(dh[0], dh[1], 5, 2, 2, outsize=(8, 8), initialW=init, nobias=True))
                layer_list['conv_p%d_2' % i] = (
                    cl.Deconvolution2D(dh[1], dh[2], 5, 2, 2, outsize=(16, 16), initialW=init, nobias=True))
                layer_list['conv_p%d_3' % i] = (
                    cl.Deconvolution2D(dh[2], dh[3], 5, 2, 2, outsize=(32, 32), initialW=init, nobias=True))
                layer_list['conv_p%d_4' % i] = (
                    cl.Deconvolution2D(dh[3], dim_out, 5, 2, 2, outsize=(64, 64), initialW=init))
                layer_list['linear_p%d_1_bn' % i] = cl.BatchNormalization(dh[0])
                layer_list['conv_p%d_1_bn' % i] = cl.BatchNormalization(dh[1])
                layer_list['conv_p%d_2_bn' % i] = cl.BatchNormalization(dh[2])
                layer_list['conv_p%d_3_bn' % i] = cl.BatchNormalization(dh[3])

            for k, v in layer_list.items():
                setattr(self, k, v)

            self.texture_base = chainer.Parameter(
                chainer.initializers.Constant(0), (3, self.texture_size, 6 * self.texture_size))

    def to_gpu(self, device=None):
        super(ConvolutionTextureDecoder, self).to_gpu(device)
        self.vertices = chainer.cuda.to_gpu(self.vertices, device)
        self.faces = chainer.cuda.to_gpu(self.faces, device)

    def compute_vertices(self):
        xp = self.xp
        vertices = []
        faces = []
        for dim1 in xp.arange(self.grid_size - 1):
            for dim2 in xp.arange(self.grid_size - 1):
                p1 = dim1, dim2
                p2 = dim1 + 1, dim2
                p3 = dim1, dim2 + 1
                p4 = dim1 + 1, dim2 + 1
                if p1 not in vertices:
                    vertices.append(p1)
                if p2 not in vertices:
                    vertices.append(p2)
                if p3 not in vertices:
                    vertices.append(p3)
                if p4 not in vertices:
                    vertices.append(p4)
                vi1 = vertices.index(p1)
                vi2 = vertices.index(p2)
                vi3 = vertices.index(p3)
                vi4 = vertices.index(p4)
                faces.append((vi1, vi2, vi4))
                faces.append((vi1, vi4, vi3))
        vertices = xp.array(vertices)
        vertices = 1. * vertices / (self.grid_size - 1) * (self.texture_size - 1)
        faces = xp.array(faces)
        num_vertices_div_6 = vertices.shape[0]
        num_faces_div_6 = faces.shape[0]

        vertices = xp.tile(vertices, (6, 1))
        vertices[:, 0] += xp.repeat(xp.arange(6) * self.texture_size, num_vertices_div_6)
        faces = xp.tile(faces, (6, 1))
        faces += xp.repeat(xp.arange(6) * num_vertices_div_6, num_faces_div_6)[:, None]

        vertices = vertices.astype('float32')
        faces = faces.astype('int32')
        self.vertices = vertices
        self.faces = faces

    def __call__(self, x):
        batch_size = x.shape[0]

        hs = []
        for i in range(6):
            linear1 = getattr(self, 'linear_p%d_1' % i)
            conv1 = getattr(self, 'conv_p%d_1' % i)
            conv2 = getattr(self, 'conv_p%d_2' % i)
            conv3 = getattr(self, 'conv_p%d_3' % i)
            conv4 = getattr(self, 'conv_p%d_4' % i)
            linear1_bn = getattr(self, 'linear_p%d_1_bn' % i)
            conv1_bn = getattr(self, 'conv_p%d_1_bn' % i)
            conv2_bn = getattr(self, 'conv_p%d_2_bn' % i)
            conv3_bn = getattr(self, 'conv_p%d_3_bn' % i)

            h = linear1(x)
            h = h.reshape((h.shape[0], -1, 4, 4))
            h = cf.relu(linear1_bn(h))
            h = cf.relu(conv1_bn(conv1(h)))
            h = cf.relu(conv2_bn(conv2(h)))
            h = cf.relu(conv3_bn(conv3(h)))
            h = conv4(h)
            hs.append(h)
        if self.symmetric:
            hs[0] = hs[1] = (hs[0] + hs[1]) / 2
            hs[2] = (hs[2] + hs[2][:, :, ::-1, :]) / 2.
            hs[3] = (hs[3] + hs[3][:, :, ::-1, :]) / 2.
            hs[4] = (hs[4] + hs[4][:, :, ::-1, :]) / 2.
            hs[5] = (hs[5] + hs[5][:, :, ::-1, :]) / 2.

        hs = cf.stack(hs)
        hs *= self.scaling
        hs = hs.transpose((1, 2, 0, 3, 4))  # [bs, 3, 6, 64, 64]
        bias = hs.reshape((batch_size, 3, -1, self.texture_size)).transpose((0, 1, 3, 2))  # [bs, 3, 64, 6 * 64]
        base = cf.tile(self.texture_base[None, :, :, :], (batch_size, 1, 1, 1))
        textures = cf.sigmoid(base + bias)
        vertices = cf.tile(self.vertices[None, :, :], (batch_size, 1, 1))

        return vertices, self.faces, textures


class DummyDecoder(chainer.Chain):
    pass


def get_shape_decoder(name, dim_in, scaling, symmetric):
    if name == 'basic':
        return BasicShapeDecoder(dim_in, scaling=scaling)
    elif name == 'basic_symmetric':
        return BasicSymmetricShapeDecoder(dim_in, scaling=scaling)
    elif name == 'conv':
        return ConvolutionShapeDecoder(dim_in, scaling=scaling, symmetric=symmetric)
    elif name == 'fc':
        return FCShapeDecoder(dim_in, scaling=scaling, symmetric=symmetric)
    elif name == 'resnet':
        return ResNetShapeDecoder(dim_in, scaling=scaling)
    elif name == 'dummy':
        return DummyDecoder()
    else:
        raise Exception


def get_texture_decoder(name, dim_in, scaling, symmetric):
    if name == 'dummy':
        return DummyDecoder()
    elif name == 'conv':
        return ConvolutionTextureDecoder(dim_in, scaling=scaling, symmetric=symmetric)
    else:
        raise Exception
