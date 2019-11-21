import chainer
import chainer.functions as cf
import chainer.links.caffe
import neural_renderer

import decoders
import discriminators
import encoders
import layers
import loss_functions
import voxelization


class ShapeNetModel(chainer.Chain):
    def __init__(
            self,
            encoder_type,
            shape_decoder_type,
            texture_decoder_type,
            discriminator_type,
            vertex_scaling,
            texture_scaling,
            silhouette_loss_type,
            silhouette_loss_levels,
            lambda_silhouettes,
            lambda_textures,
            lambda_perceptual,
            lambda_inflation,
            lambda_discriminator,
            lambda_graph_laplacian,
            lambda_edge_length,
            single_view_training,
            class_conditional,
            iterative_optimization,
            discriminator_mode,
            no_texture,
            symmetric,
            num_views,
            dim_hidden=512,
            anti_aliasing=False,
    ):
        super(ShapeNetModel, self).__init__()
        self.trainer = None
        self.dataset_name = 'shapenet'

        # model size
        self.dim_hidden = dim_hidden

        # loss weights
        self.silhouette_loss_type = silhouette_loss_type
        self.silhouette_loss_levels = silhouette_loss_levels
        self.lambda_silhouettes = lambda_silhouettes
        self.lambda_textures = lambda_textures
        self.lambda_perceptual = lambda_perceptual
        self.lambda_discriminator = lambda_discriminator
        self.lambda_inflation = lambda_inflation
        self.lambda_graph_laplacian = lambda_graph_laplacian
        self.lambda_edge_length = lambda_edge_length

        # others
        self.single_view_training = single_view_training
        self.class_conditional = class_conditional
        self.iterative_optimization = iterative_optimization
        self.discriminator_mode = discriminator_mode
        self.no_texture = no_texture
        self.num_views = num_views
        self.use_depth = False
        self.symmetric = symmetric

        # setup renderer
        self.renderer = neural_renderer.Renderer()
        self.renderer.image_size = 224
        self.renderer.anti_aliasing = anti_aliasing
        self.renderer.perspective = True
        self.renderer.viewing_angle = self.xp.degrees(self.xp.arctan(16. / 60.))
        self.renderer.camera_mode = 'look_at'
        self.renderer.blur_size = 0

        with self.init_scope():
            # setup links
            dim_in_encoder = 3
            if no_texture:
                texture_decoder_type = 'dummy'
                dim_in_discriminator = 1
            else:
                dim_in_discriminator = 4

            self.encoder = encoders.get_encoder(encoder_type, dim_in_encoder, self.dim_hidden)
            self.shape_decoder = decoders.get_shape_decoder(
                shape_decoder_type, self.dim_hidden, vertex_scaling, self.symmetric)
            self.texture_decoder = decoders.get_texture_decoder(
                texture_decoder_type, self.dim_hidden, texture_scaling, self.symmetric)
            self.discriminator = discriminators.get_discriminator(discriminator_type, dim_in_discriminator)
            self.shape_encoder = self.encoder

    def encode(self, images):
        return self.encoder(images[:, :3])

    def decode_shape(self, codes):
        vertices, faces = self.shape_decoder(codes)
        vertices *= 0.5
        return vertices, faces

    def decode_texture(self, codes):
        vertices_t, faces_t, textures = self.texture_decoder(codes)
        return vertices_t, faces_t, textures

    def rotate(self, vertices, viewpoints):
        self.renderer.viewpoints = viewpoints
        return vertices

    def render(self, vertices, faces, viewpoints, vertices_t=None, faces_t=None, textures=None):
        vertices = self.rotate(vertices, viewpoints)

        if vertices_t is None:
            images = self.renderer.render_silhouettes(vertices, faces)
            images = images[:, None, :, :]
        else:
            images = self.renderer.render(vertices, faces, vertices_t, faces_t, textures)

        return images

    def predict_and_render(self, images, rotation_matrices):
        codes = self.encode(images)
        vertices, faces = self.decode_shape(codes)
        if self.no_texture:
            images = self.render(vertices, faces, rotation_matrices)
        else:
            vertices_t, faces_t, textures = self.decode_texture(codes)
            images = self.render(vertices, faces, rotation_matrices, vertices_t, faces_t, textures)

        return images

    def evaluate_iou(self, images, voxels):
        codes = self.encode(images)
        vertices, faces = self.decode_shape(codes)
        faces = vertices[:, faces].data
        faces += 0.5  # normalization
        voxels_predicted = voxelization.voxelize(faces, 32)[:, :, :, ::-1]

        iou = (voxels * voxels_predicted).sum((1, 2, 3)) / (0 < (voxels + voxels_predicted)).sum((1, 2, 3))
        return iou

    def compute_loss_one_view(self, images_in, viewpoints, labels):
        # predict shapes and render them
        codes_in = self.encode(images_in)
        vertices, faces = self.decode_shape(codes_in)

        if self.no_texture:
            images_out = self.render(vertices, faces, viewpoints)
        else:
            vertices_t, faces_t, textures = self.decode_texture(codes_in)
            images_out = self.render(vertices, faces, viewpoints, vertices_t, faces_t, textures)

        # compute loss
        loss_gen = chainer.Variable(self.xp.array(0, 'float32'))
        loss_dis = chainer.Variable(self.xp.array(0, 'float32'))
        loss_list = {}

        loss_silhouettes = loss_functions.silhouette_loss(
            images_in[:, -1], images_out[:, -1], self.silhouette_loss_levels)
        loss_gen += self.lambda_silhouettes * loss_silhouettes
        loss_list['loss_silhouettes'] = loss_silhouettes

        if (not self.no_texture) and self.lambda_perceptual != 0:
            loss_perceptual = loss_functions.perceptual_texture_loss(images_in[:, :3], images_out[:, :3])
            loss_gen += self.lambda_perceptual * loss_perceptual
            loss_list['loss_perceptual'] = loss_perceptual

        if self.lambda_discriminator != 0:
            # set real images
            if self.discriminator_mode == 'seen_unseen':
                images_real = images_out
            elif self.discriminator_mode == 'real_rendered':
                if self.no_texture:
                    images_real = images_in[:, -1:].copy()
                else:
                    images_real = images_in.copy()
                    mask = images_real[:, -1]
                    images_rgb = 0.5 * images_real[:, :3] + 0.5
                    images_real[:, :3] = mask[:, None, :, :] * images_rgb

            # set fake images
            # render from random viewpoints
            if self.no_texture:
                images_fake = self.render(vertices, faces, viewpoints[::-1])
            else:
                images_fake = self.render(vertices, faces, viewpoints[::-1], vertices_t, faces_t, textures)

            # set labels
            if not self.class_conditional:
                labels = None

            if self.iterative_optimization:
                # print images_real.min(), images_real.max()
                # print images_fake.data.min(), images_fake.data.max()
                # print images_real.shape, images_fake.shape
                discrimination_real = self.discriminator(images_real, viewpoints, labels)
                discrimination_fake = self.discriminator(images_fake, viewpoints[::-1], labels)

                loss_discriminator_adv = (
                        loss_functions.adversarial_loss(discrimination_real, 'kl_fake') +
                        loss_functions.adversarial_loss(discrimination_fake, 'kl_real'))
                loss_gen += self.lambda_discriminator * loss_discriminator_adv
                loss_list['loss_discriminator_adv'] = loss_discriminator_adv

                if isinstance(images_real, chainer.Variable):
                    images_real = images_real.data
                if isinstance(images_fake, chainer.Variable):
                    images_fake = images_fake.data
                discrimination_real = self.discriminator(images_real, viewpoints, labels)
                discrimination_fake = self.discriminator(images_fake, viewpoints[::-1], labels)

                loss_discriminator = (
                        loss_functions.adversarial_loss(discrimination_real, 'kl_real') +
                        loss_functions.adversarial_loss(discrimination_fake, 'kl_fake'))
                loss_dis += loss_discriminator
                loss_list['loss_discriminator'] = loss_discriminator
            else:
                images_real = layers.invert_gradient(images_real, self.lambda_discriminator)
                images_fake = layers.invert_gradient(images_fake, self.lambda_discriminator)

                discrimination_real = self.discriminator(images_real, viewpoints, labels)
                discrimination_fake = self.discriminator(images_fake, viewpoints[::-1], labels)
                loss_discriminator = (
                        loss_functions.adversarial_loss(discrimination_real, 'kl_real') +
                        loss_functions.adversarial_loss(discrimination_fake, 'kl_fake'))
                loss_gen += loss_discriminator
                loss_list['loss_discriminator'] = loss_discriminator

        if self.lambda_inflation != 0:
            loss_inflation = (
                loss_functions.inflation_loss(vertices, self.shape_decoder.faces, self.shape_decoder.degrees))
            loss_gen += self.lambda_inflation * loss_inflation
            loss_list['loss_inflation'] = loss_inflation

        if self.lambda_graph_laplacian != 0:
            loss_graph_laplacian = (
                loss_functions.graph_laplacian_loss(vertices, self.shape_decoder.laplacian))
            loss_gen += self.lambda_graph_laplacian * loss_graph_laplacian
            loss_list['loss_graph_laplacian'] = loss_graph_laplacian

        if self.lambda_edge_length != 0:
            loss_edge_length = loss_functions.edge_length_loss(vertices, self.shape_decoder.faces)
            loss_gen += self.lambda_edge_length * loss_edge_length
            loss_list['loss_edge_length'] = loss_edge_length

        chainer.reporter.report(loss_list, self)

        if self.iterative_optimization:
            return loss_gen, loss_dis
        else:
            return loss_gen

    def compute_loss_multi_view(self, images_in_a, images_in_b, viewpoints_a, viewpoints_b, labels):
        # predict shapes and render them
        batch_size = images_in_a.shape[0]
        images_in = cf.concat((images_in_a, images_in_b), axis=0)
        codes_in = self.encode(images_in)
        vertices, faces = self.decode_shape(codes_in)
        vertices_a, vertices_b = vertices[:batch_size], vertices[batch_size:]

        if self.no_texture:
            images_out_a_a = self.render(vertices_a, faces, viewpoints_a)
            images_out_a_b = self.render(vertices_a, faces, viewpoints_b)
            images_out_b_a = self.render(vertices_b, faces, viewpoints_a)
            images_out_b_b = self.render(vertices_b, faces, viewpoints_b)
        else:
            vertices_t, faces_t, textures = self.decode_texture(codes_in)
            vertices_t = vertices_t[:batch_size]
            textures_a = textures[:batch_size]
            textures_b = textures[batch_size:]
            images_out_a_a = self.render(vertices_a, faces, viewpoints_a, vertices_t, faces_t, textures_a)
            images_out_a_b = self.render(vertices_a, faces, viewpoints_b, vertices_t, faces_t, textures_a)
            images_out_b_a = self.render(vertices_b, faces, viewpoints_a, vertices_t, faces_t, textures_b)
            images_out_b_b = self.render(vertices_b, faces, viewpoints_b, vertices_t, faces_t, textures_b)

        # compute loss
        loss_gen = chainer.Variable(self.xp.array(0, 'float32'))
        loss_dis = chainer.Variable(self.xp.array(0, 'float32'))
        loss_list = {}

        if self.silhouette_loss_type == 'l2':
            s_in_a = images_in_a[:, -1]
            s_in_b = images_in_b[:, -1]
            loss_silhouettes = (
                loss_functions.silhouette_loss(s_in_a, images_out_a_a[:, -1], self.silhouette_loss_levels) +
                loss_functions.silhouette_loss(s_in_a, images_out_b_a[:, -1], self.silhouette_loss_levels) +
                loss_functions.silhouette_loss(s_in_b, images_out_a_b[:, -1], self.silhouette_loss_levels) +
                loss_functions.silhouette_loss(s_in_b, images_out_b_b[:, -1], self.silhouette_loss_levels))
        elif self.silhouette_loss_type == 'iou':
            loss_silhouettes = (
                loss_functions.silhouette_iou_loss(images_in_a[:, -1], images_out_a_a[:, -1]) +
                loss_functions.silhouette_iou_loss(images_in_a[:, -1], images_out_b_a[:, -1]) +
                loss_functions.silhouette_iou_loss(images_in_b[:, -1], images_out_a_b[:, -1]) +
                loss_functions.silhouette_iou_loss(images_in_b[:, -1], images_out_b_b[:, -1]))
        elif self.silhouette_loss_type == 'l2u':
            loss_silhouettes = (
                cf.sum(cf.square(images_in_a[:, -1] - images_out_a_a[:, -1])) / batch_size +
                cf.sum(cf.square(images_in_a[:, -1] - images_out_b_a[:, -1])) / batch_size +
                cf.sum(cf.square(images_in_b[:, -1] - images_out_a_b[:, -1])) / batch_size +
                cf.sum(cf.square(images_in_b[:, -1] - images_out_b_b[:, -1])) / batch_size)
        loss_gen += self.lambda_silhouettes * loss_silhouettes
        loss_list['loss_silhouettes'] = loss_silhouettes

        if (not self.no_texture) and self.lambda_textures != 0:
            loss_textures = (
                    loss_functions.loss_textures(images_in_a[:, :3], images_out_a_a[:, :3],
                                                 self.silhouette_loss_levels) +
                    loss_functions.loss_textures(images_in_a[:, :3], images_out_b_a[:, :3],
                                                 self.silhouette_loss_levels) +
                    loss_functions.loss_textures(images_in_b[:, :3], images_out_a_b[:, :3],
                                                 self.silhouette_loss_levels) +
                    loss_functions.loss_textures(images_in_b[:, :3], images_out_b_b[:, :3],
                                                 self.silhouette_loss_levels))
            loss_gen += self.lambda_textures * loss_textures
            loss_list['loss_textures'] = loss_textures

        if (not self.no_texture) and self.lambda_perceptual != 0:
            loss_perceptual = (
                    loss_functions.perceptual_texture_loss(images_in_a[:, :3], images_out_a_a[:, :3]) +
                    loss_functions.perceptual_texture_loss(images_in_a[:, :3], images_out_b_a[:, :3]) +
                    loss_functions.perceptual_texture_loss(images_in_b[:, :3], images_out_a_b[:, :3]) +
                    loss_functions.perceptual_texture_loss(images_in_b[:, :3], images_out_b_b[:, :3]))
            loss_gen += self.lambda_perceptual * loss_perceptual
            loss_list['loss_perceptual'] = loss_perceptual

        if self.lambda_discriminator != 0:
            # set real images
            if self.discriminator_mode == 'seen_unseen':
                images_real_a = images_out_a_a
                images_real_b = images_out_b_b
            elif self.discriminator_mode == 'real_rendered':
                images_real_a = images_in_a.copy()
                mask_a = images_real_a[:, -1]
                images_rgb_a = 0.5 * images_real_a[:, :3] + 0.5
                images_real_a[:, :3] = mask_a[:, None, :, :] * images_rgb_a
                images_real_b = images_in_b.copy()
                mask_b = images_real_b[:, -1]
                images_rgb_b = 0.5 * images_real_b[:, :3] + 0.5
                images_real_b[:, :3] = mask_b[:, None, :, :] * images_rgb_b
            images_real = cf.concat((images_real_a, images_real_b), axis=0)
            viewpoints_real = cf.concat((viewpoints_a, viewpoints_b), axis=0)

            # set fake images
            # render from other viewpoints (fake image)
            if self.no_texture:
                images_fake_a = self.render(vertices_a, faces, viewpoints_a[::-1])
                images_fake_b = self.render(vertices_b, faces, viewpoints_b[::-1])
            else:
                images_fake_a = self.render(vertices_a, faces, viewpoints_a[::-1], vertices_t, faces_t, textures_a)
                images_fake_b = self.render(vertices_b, faces, viewpoints_b[::-1], vertices_t, faces_t, textures_b)
            images_fake = cf.concat((images_fake_a, images_fake_b), axis=0)
            viewpoints_fake = cf.concat((viewpoints_a[::-1], viewpoints_b[::-1]), axis=0)

            # set labels
            if not self.class_conditional:
                labels_real = None
                labels_fake = None
            else:
                labels_real = self.xp.concatenate((labels, labels), axis=0)
                labels_fake = self.xp.concatenate((labels, labels), axis=0)

            if self.iterative_optimization:
                discrimination_real = self.discriminator(images_real, viewpoints_real, labels_real)
                discrimination_fake = self.discriminator(images_fake, viewpoints_fake, labels_fake)

                loss_discriminator_adv = (
                        loss_functions.adversarial_loss(discrimination_real, 'kl_fake') +
                        loss_functions.adversarial_loss(discrimination_fake, 'kl_real'))
                loss_gen += self.lambda_discriminator * loss_discriminator_adv
                loss_list['loss_discriminator_adv'] = loss_discriminator_adv

                if isinstance(images_real, chainer.Variable):
                    images_real = images_real.data
                if isinstance(images_fake, chainer.Variable):
                    images_fake = images_fake.data
                discrimination_real = self.discriminator(images_real, viewpoints_real, labels_real)
                discrimination_fake = self.discriminator(images_fake, viewpoints_fake, labels_fake)

                loss_discriminator = (
                        loss_functions.adversarial_loss(discrimination_real, 'kl_real') +
                        loss_functions.adversarial_loss(discrimination_fake, 'kl_fake'))
                loss_dis += loss_discriminator
                loss_list['loss_discriminator'] = loss_discriminator
            else:
                images_real = layers.invert_gradient(images_real, self.lambda_discriminator)
                images_fake = layers.invert_gradient(images_fake, self.lambda_discriminator)
                discrimination_real = self.discriminator(images_real, viewpoints_real, labels_real)
                discrimination_fake = self.discriminator(images_fake, viewpoints_fake, labels_fake)

                loss_discriminator = (
                        loss_functions.adversarial_loss(discrimination_real, 'kl_real') +
                        loss_functions.adversarial_loss(discrimination_fake, 'kl_fake'))
                loss_gen += loss_discriminator
                loss_list['loss_discriminator'] = loss_discriminator

        if self.lambda_inflation != 0:
            loss_inflation = (
                    loss_functions.inflation_loss(vertices_a, self.shape_decoder.faces, self.shape_decoder.degrees) +
                    loss_functions.inflation_loss(vertices_b, self.shape_decoder.faces, self.shape_decoder.degrees))
            loss_gen += self.lambda_inflation * loss_inflation
            loss_list['loss_inflation'] = loss_inflation

        if self.lambda_graph_laplacian != 0:
            loss_graph_laplacian = (
                    loss_functions.graph_laplacian_loss(vertices_a, self.shape_decoder.laplacian) +
                    loss_functions.graph_laplacian_loss(vertices_b, self.shape_decoder.laplacian))
            loss_gen += self.lambda_graph_laplacian * loss_graph_laplacian
            loss_list['loss_graph_laplacian'] = loss_graph_laplacian

        if self.lambda_edge_length != 0:
            loss_edge_length = (
                    loss_functions.edge_length_loss(vertices_a, self.shape_decoder.faces) +
                    loss_functions.edge_length_loss(vertices_b, self.shape_decoder.faces))
            loss_gen += self.lambda_edge_length * loss_edge_length
            loss_list['loss_edge_length'] = loss_edge_length

        chainer.reporter.report(loss_list, self)

        if self.iterative_optimization:
            return loss_gen, loss_dis
        else:
            return loss_gen

    def __call__(self, images_in_a, images_in_b, viewpoints_a, viewpoints_b, labels):
        if self.single_view_training:
            return self.compute_loss_one_view(images_in_a, viewpoints_a, labels)
        else:
            return self.compute_loss_multi_view(images_in_a, images_in_b, viewpoints_a, viewpoints_b, labels)


class PascalModel(chainer.Chain):
    def __init__(
            self,
            encoder_type,
            shape_decoder_type,
            texture_decoder_type,
            discriminator_type,
            silhouette_loss_type,
            vertex_scaling,
            texture_scaling,
            silhouette_loss_levels,
            lambda_silhouettes,
            lambda_perceptual,
            lambda_inflation,
            lambda_graph_laplacian,
            lambda_discriminator,
            no_texture,
            class_conditional,
            symmetric,
            dim_hidden=512,
            image_size=224,
            anti_aliasing=False,
    ):
        super(PascalModel, self).__init__()
        self.trainer = None
        self.dataset_name = 'pascal'

        # model size
        self.dim_hidden = dim_hidden

        # loss type
        self.silhouette_loss_type = silhouette_loss_type
        self.silhouette_loss_levels = silhouette_loss_levels

        # loss weights
        self.lambda_silhouettes = lambda_silhouettes
        self.lambda_perceptual = lambda_perceptual
        self.lambda_discriminator = lambda_discriminator
        self.lambda_inflation = lambda_inflation
        self.lambda_graph_laplacian = lambda_graph_laplacian

        # others
        self.no_texture = no_texture
        self.class_conditional = class_conditional
        self.symmetric = symmetric
        self.use_depth = False

        # setup renderer
        self.renderer = neural_renderer.Renderer()
        self.renderer.image_size = image_size
        self.renderer.anti_aliasing = anti_aliasing
        self.renderer.perspective = False
        self.renderer.camera_mode = 'none'

        with self.init_scope():
            # setup links
            dim_in_encoder = 3
            if no_texture:
                texture_decoder_type = 'dummy'
                dim_in_discriminator = 1
            else:
                dim_in_discriminator = 4

            self.encoder = encoders.get_encoder(encoder_type, dim_in_encoder, self.dim_hidden)
            self.shape_decoder = decoders.get_shape_decoder(
                shape_decoder_type, self.dim_hidden, vertex_scaling, symmetric)
            self.texture_decoder = decoders.get_texture_decoder(
                texture_decoder_type, self.dim_hidden, texture_scaling, self.symmetric)
            self.discriminator = discriminators.get_discriminator(discriminator_type, dim_in_discriminator)

    def encode(self, images):
        return self.encoder(images[:, :3])

    def decode_shape(self, codes):
        vertices, faces = self.shape_decoder(codes)
        return vertices, faces

    def decode_texture(self, codes):
        vertices_t, faces_t, textures = self.texture_decoder(codes)
        return vertices_t, faces_t, textures

    def predict_shape(self, images):
        vertices, faces = self.shape_decoder(self.encoder(images))
        return vertices, faces

    def predict_texture(self, images):
        vertices_t, faces_t, textures = self.texture_decoder(self.encoder(images))
        return vertices_t, faces_t, textures

    def rotate(self, vertices, rotation_matrices):
        # viewpoint transformation
        rm, v = cf.broadcast(rotation_matrices[:, None, :, :], vertices[:, :, None, :])
        vertices = cf.sum(rm * v, axis=-1)

        # adjust z-axes for rendering. [-1, 1] -> [0 + eps, 2]
        bias = self.xp.zeros_like(vertices)
        bias[:, :, -1] = 1 + 1e-5
        vertices += bias

        return vertices

    def render(self, vertices, faces, rotation_matrices, vertices_t=None, faces_t=None, textures=None):
        vertices = self.rotate(vertices, rotation_matrices)

        if vertices_t is None:
            images = self.renderer.render_silhouettes(vertices, faces)
            images = images[:, None, :, :]
        else:
            images = self.renderer.render(vertices, faces, vertices_t, faces_t, textures)

        return images

    def predict_and_render(self, images, rotation_matrices):
        codes = self.encode(images)
        vertices, faces = self.decode_shape(codes)
        if self.no_texture:
            vertices, faces = self.predict_shape(images)
            images = self.render(vertices, faces, rotation_matrices)
        else:
            vertices_t, faces_t, textures = self.predict_texture(images)
            images = self.render(vertices, faces, rotation_matrices, vertices_t, faces_t, textures)
        return images

    def evaluate_iou(self, images, voxels):
        vertices, faces = self.predict_shape(images)
        faces = vertices[:, faces].data
        faces = (faces + 1.0) * 0.5  # normalization
        voxels_predicted = voxelization.voxelize(faces, 32)
        voxels_predicted = voxels_predicted.transpose((0, 1, 3, 2))[:, ::-1, ::-1, ::-1]
        iou = (voxels * voxels_predicted).sum((1, 2, 3)) / (0 < (voxels + voxels_predicted)).sum((1, 2, 3))
        return iou

    def __call__(self, images_in, images_ref, rotation_matrices, rotation_matrices_random, labels):
        """
        Compute loss to be optimized.

        Args:
            images_in (cupy.ndarray): Images for encoders [batch_size, 3, 224, 224].
            images_ref (cupy.ndarray): Target color maps and silhouettes [batch_size, 3, 224, 224].
            rotation_matrices: Matrices for view point transformation.

        Returns:
            chainer.Variable: Loss.
        """

        # predict shapes and render them
        codes_in = self.encode(images_in)
        vertices, faces = self.decode_shape(codes_in)

        if self.no_texture:
            images_out = self.render(vertices, faces, rotation_matrices)
        else:
            vertices_t, faces_t, textures = self.decode_texture(codes_in)
            images_out = self.render(vertices, faces, rotation_matrices, vertices_t, faces_t, textures)

        # compute loss
        loss = chainer.Variable(self.xp.array(0, 'float32'))
        loss_list = {}

        if self.silhouette_loss_type == 'l2':
            loss_silhouettes = loss_functions.silhouette_loss(
                images_ref[:, -1], images_out[:, -1], self.silhouette_loss_levels)
        else:
            loss_silhouettes = loss_functions.silhouette_iou_loss(images_ref[:, -1], images_out[:, -1])
        loss += self.lambda_silhouettes * loss_silhouettes
        loss_list['loss_silhouettes'] = loss_silhouettes

        if (not self.no_texture) and self.lambda_perceptual != 0:
            loss_perceptual = loss_functions.perceptual_texture_loss(images_ref, images_out, zero_mean=False)
            loss += self.lambda_perceptual * loss_perceptual
            loss_list['loss_perceptual'] = loss_perceptual

        if self.lambda_discriminator != 0:
            # render from other viewpoints (fake image)
            if self.no_texture:
                images_fake = self.render(vertices, faces, rotation_matrices_random)
            else:
                images_fake = self.render(vertices, faces, rotation_matrices_random, vertices_t, faces_t, textures)

            images_real = layers.invert_gradient(images_out, self.lambda_discriminator)
            images_fake = layers.invert_gradient(images_fake, self.lambda_discriminator)
            viewpoints_real = rotation_matrices
            viewpoints_fake = rotation_matrices_random

            if self.class_conditional:
                discrimination_real = self.discriminator(images_real, viewpoints_real, labels)
                discrimination_fake = self.discriminator(images_fake, viewpoints_fake, labels)
            else:
                discrimination_real = self.discriminator(images_real, viewpoints_real)
                discrimination_fake = self.discriminator(images_fake, viewpoints_fake)

            loss_discriminator = (
                    loss_functions.adversarial_loss(discrimination_real, 'kl_real') +
                    loss_functions.adversarial_loss(discrimination_fake, 'kl_fake'))
            loss += loss_discriminator
            loss_list['loss_discriminator'] = loss_discriminator

        if self.lambda_inflation != 0:
            loss_inflation = loss_functions.inflation_loss(
                vertices, self.shape_decoder.faces, self.shape_decoder.degrees)
            loss += self.lambda_inflation * loss_inflation
            loss_list['loss_inflation'] = loss_inflation

        if self.lambda_graph_laplacian != 0:
            loss_graph_laplacian = loss_functions.graph_laplacian_loss(vertices, self.shape_decoder.laplacian)
            loss += self.lambda_graph_laplacian * loss_graph_laplacian
            loss_list['loss_graph_laplacian'] = loss_graph_laplacian


        chainer.reporter.report(loss_list, self)

        return loss
