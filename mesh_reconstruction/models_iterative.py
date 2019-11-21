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
            silhouette_loss_levels,
            lambda_silhouettes,
            lambda_textures,
            lambda_perceptual,
            lambda_inflation,
            lambda_discriminator,
            lambda_discriminator2,
            lambda_graph_laplacian,
            single_view_training,
            class_conditional,
            no_texture,
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
        self.silhouette_loss_levels = silhouette_loss_levels
        self.lambda_silhouettes = lambda_silhouettes
        self.lambda_textures = lambda_textures
        self.lambda_perceptual = lambda_perceptual
        self.lambda_discriminator = lambda_discriminator
        self.lambda_discriminator2 = lambda_discriminator2
        self.lambda_inflation = lambda_inflation
        self.lambda_graph_laplacian = lambda_graph_laplacian

        # others
        self.single_view_training = single_view_training
        self.class_conditional = class_conditional
        self.no_texture = no_texture
        self.num_views = num_views
        self.use_depth = False

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
            self.shape_decoder = decoders.get_shape_decoder(shape_decoder_type, self.dim_hidden, vertex_scaling)
            self.texture_decoder = decoders.get_texture_decoder(texture_decoder_type, self.dim_hidden, texture_scaling)
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

    def compute_loss_one_view_discriminator(self, images_in, viewpoints, labels):
        # predict shapes and render them
        codes_in = self.encode(images_in)
        vertices, faces = self.decode_shape(codes_in)

        if self.no_texture:
            images_out = self.render(vertices, faces, viewpoints).data
        else:
            vertices_t, faces_t, textures = self.decode_texture(codes_in)
            images_out = self.render(vertices, faces, viewpoints, vertices_t, faces_t, textures).data

        if self.no_texture:
            images_fake = self.render(vertices, faces, viewpoints[::-1]).data
        else:
            images_fake = self.render(vertices, faces, viewpoints[::-1], vertices_t, faces_t, textures).data

        # compute loss
        loss = chainer.Variable(self.xp.array(0, 'float32'))
        loss_list = {}
        if self.lambda_discriminator != 0:
            images_real2 = images_out.copy()
            images_fake2 = images_fake.copy()

            if self.class_conditional:
                discrimination_real = self.discriminator(images_real2, viewpoints, labels)
                discrimination_fake = self.discriminator(images_fake2, viewpoints[::-1], labels)
            else:
                discrimination_real = self.discriminator(images_real2, viewpoints)
                discrimination_fake = self.discriminator(images_fake2, viewpoints[::-1])

            loss_discriminator = (
                loss_functions.adversarial_loss(discrimination_real, 'kl_real') +
                loss_functions.adversarial_loss(discrimination_fake, 'kl_fake'))
            loss += loss_discriminator
            loss_list['loss_discriminator'] = loss_discriminator

        if self.lambda_discriminator2 != 0:
            images_real2 = images_in.copy()
            mask = images_real[:, -1]
            images_rgb = 0.5 * images_real2[:, :3] + 0.5
            images_real2[:, :3] = mask[:, None, :, :] * images_rgb
            images_fake2 = images_fake.copy()

            if self.class_conditional:
                discrimination_real = self.discriminator(images_real2, viewpoints, labels)
                discrimination_fake = self.discriminator(images_fake2, viewpoints[::-1], labels)
            else:
                discrimination_real = self.discriminator(images_real2, viewpoints)
                discrimination_fake = self.discriminator(images_fake2, viewpoints[::-1])

            loss_discriminator2 = (
                loss_functions.adversarial_loss(discrimination_real, 'kl_real') +
                loss_functions.adversarial_loss(discrimination_fake, 'kl_fake'))
            loss += loss_discriminator2
            loss_list['loss_discriminator2'] = loss_discriminator2

        chainer.reporter.report(loss_list, self)

        return loss

    def compute_loss_one_view_generator(self, images_in, viewpoints, labels):
        # predict shapes and render them
        codes_in = self.encode(images_in)
        vertices, faces = self.decode_shape(codes_in)

        if self.no_texture:
            images_out = self.render(vertices, faces, viewpoints)
        else:
            vertices_t, faces_t, textures = self.decode_texture(codes_in)
            images_out = self.render(vertices, faces, viewpoints, vertices_t, faces_t, textures)

        # compute loss
        loss = chainer.Variable(self.xp.array(0, 'float32'))
        loss_list = {}

        loss_silhouettes = loss_functions.silhouette_loss(
            images_in[:, -1], images_out[:, -1], self.silhouette_loss_levels)
        loss += self.lambda_silhouettes * loss_silhouettes
        loss_list['loss_silhouettes'] = loss_silhouettes

        if (not self.no_texture) and self.lambda_perceptual != 0:
            loss_perceptual = loss_functions.perceptual_texture_loss(images_in[:, :3], images_out[:, :3])
            loss += self.lambda_perceptual * loss_perceptual
            loss_list['loss_perceptual'] = loss_perceptual

        if self.lambda_discriminator != 0:
            # render from other viewpoints (fake image)
            if self.no_texture:
                images_fake = self.render(vertices, faces, viewpoints[::-1])
            else:
                images_fake = self.render(vertices, faces, viewpoints[::-1], vertices_t, faces_t, textures)

            images_real = images_out

            if self.class_conditional:
                discrimination_real = self.discriminator(images_real, viewpoints, labels)
                discrimination_fake = self.discriminator(images_fake, viewpoints[::-1], labels)
            else:
                discrimination_real = self.discriminator(images_real, viewpoints)
                discrimination_fake = self.discriminator(images_fake, viewpoints[::-1])

            loss_discriminator = (
                loss_functions.adversarial_loss(discrimination_real, 'kl_fake') +
                loss_functions.adversarial_loss(discrimination_fake, 'kl_real'))
            loss += self.lambda_discriminator * loss_discriminator
            loss_list['loss_discriminator_adv'] = loss_discriminator

        if self.lambda_discriminator2 != 0:
            # render from other viewpoints (fake image)
            if self.no_texture:
                images_fake = self.render(vertices, faces, viewpoints[::-1])
            else:
                images_fake = self.render(vertices, faces, viewpoints[::-1], vertices_t, faces_t, textures)

            images_real = images_in.copy()
            mask = images_real[:, -1]
            images_rgb = 0.5 * images_real[:, :3] + 0.5
            images_real[:, :3] = mask[:, None, :, :] * images_rgb

            if self.class_conditional:
                discrimination_real = self.discriminator(images_real, viewpoints, labels)
                discrimination_fake = self.discriminator(images_fake, viewpoints[::-1], labels)
            else:
                discrimination_real = self.discriminator(images_real, viewpoints)
                discrimination_fake = self.discriminator(images_fake, viewpoints[::-1])

            loss_discriminator2 = (
                loss_functions.adversarial_loss(discrimination_real, 'kl_fake') +
                loss_functions.adversarial_loss(discrimination_fake, 'kl_real'))
            loss += self.lambda_discriminator2 * loss_discriminator2
            loss_list['loss_discriminator2_adv'] = loss_discriminator2

        if self.lambda_inflation != 0:
            loss_inflation = (
                loss_functions.inflation_loss(vertices, self.shape_decoder.faces, self.shape_decoder.degrees))
            loss += self.lambda_inflation * loss_inflation
            loss_list['loss_inflation'] = loss_inflation

        if self.lambda_graph_laplacian != 0:
            loss_graph_laplacian = (
                loss_functions.graph_laplacian_loss(vertices, self.shape_decoder.laplacian))
            loss += self.lambda_graph_laplacian * loss_graph_laplacian
            loss_list['loss_graph_laplacian'] = loss_graph_laplacian

        chainer.reporter.report(loss_list, self)

        return loss
