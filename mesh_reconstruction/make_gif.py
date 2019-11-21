import argparse
import glob
import numpy as np
import os
import random
import scipy.misc
import subprocess
import sys

import chainer
import cupy as cp
import neural_renderer

import dataset_pascal
import dataset_shapenet
import models


def to_rotation_matrix(elevation, azimuth):
    elevation = -np.radians(elevation)
    azimuth = -np.radians(azimuth)
    r0 = np.eye(3)
    r0[1, 1] = -1
    r1 = np.array([
        [np.cos(azimuth), 0, -np.sin(azimuth)],
        [0, 1, 0],
        [np.sin(azimuth), 0, np.cos(azimuth)],
    ])
    # r1 = np.eye(3)
    r2 = np.array([
        [1, 0, 0],
        [0, np.cos(elevation), -np.sin(elevation)],
        [0, np.sin(elevation), np.cos(elevation)],
    ])
    # r2 = np.eye(3)
    r = np.dot(r2, np.dot(r1, r0))
    return r


def run():
    class_list_shapenet = [
        '02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649',
        '03691459', '04090263', '04256520', '04379243', '04401088', '04530566']

    # system
    parser = argparse.ArgumentParser()
    parser.add_argument('-eid', '--experiment_id', type=str, required=True)
    parser.add_argument('-md', '--model_directory', type=str, default='./data/models')
    parser.add_argument('-dd', '--dataset_directory', type=str, default='./data/dataset')
    parser.add_argument('-fr', '--frame_rate', type=int, default=10)
    parser.add_argument('-rs', '--random_seed', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)

    # training
    parser.add_argument('-nt', '--no_texture', type=int, default=1)

    if 'shapenet' in sys.argv:
        # shapenet
        parser.add_argument('-ds', '--dataset', type=str, default='shapnet')
        parser.add_argument('-oid', '--object_id', type=str, required=True)
        parser.add_argument('-vid', '--view_id', type=int, required=True)

        # components
        parser.add_argument('-et', '--encoder_type', type=str, default='resnet18')
        parser.add_argument('-sdt', '--shape_decoder_type', type=str, default='conv')
        parser.add_argument('-tdt', '--texture_decoder_type', type=str, default='conv')
        parser.add_argument('-dt', '--discriminator_type', type=str, default='shapenet_patch')
        parser.add_argument('-vs', '--vertex_scaling', type=float, default=0.01)
        parser.add_argument('-ts', '--texture_scaling', type=float, default=1)
        parser.add_argument('-sym', '--symmetric', type=int, default=0)
    elif 'pascal' in sys.argv:
        # dataset
        parser.add_argument('-ds', '--dataset', type=str, default='pascal')
        parser.add_argument('-cid', '--class_id', type=str, required=True)
        parser.add_argument('-in', '--image_number', type=int, required=True)

        # components
        parser.add_argument('-et', '--encoder_type', type=str, default='resnet18pt')
        parser.add_argument('-sdt', '--shape_decoder_type', type=str, default='fc')
        parser.add_argument('-tdt', '--texture_decoder_type', type=str, default='conv')
        parser.add_argument('-dt', '--discriminator_type', type=str, default='pascal_patch')
        parser.add_argument('-sym', '--symmetric', type=int, default=1)
        parser.add_argument('-vs', '--vertex_scaling', type=float, default=0.1)
        parser.add_argument('-ts', '--texture_scaling', type=float, default=1)

    args = parser.parse_args()
    directory_model = os.path.join(args.model_directory, args.experiment_id)

    image_size = 224
    sec_rotation_a = 1
    sec_rotation_b = 3
    elevation_target = 30
    num_rotation_a = int(args.frame_rate * sec_rotation_a)
    num_rotation_b = int(args.frame_rate * sec_rotation_b)

    # set random seed, gpu
    chainer.cuda.get_device(args.gpu).use()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    cp.random.seed(args.random_seed)

    # setup model & optimizer
    if 'shapenet' in sys.argv:
        model = models.ShapeNetModel(
            encoder_type=args.encoder_type,
            shape_decoder_type=args.shape_decoder_type,
            texture_decoder_type=args.texture_decoder_type,
            discriminator_type=args.discriminator_type,
            vertex_scaling=args.vertex_scaling,
            texture_scaling=args.texture_scaling,
            silhouette_loss_levels=0,
            lambda_silhouettes=0,
            lambda_textures=0,
            lambda_perceptual=0,
            lambda_inflation=0,
            lambda_discriminator=0,
            lambda_graph_laplacian=0,
            lambda_edge_length=0,
            single_view_training=False,
            class_conditional=False,
            iterative_optimization=False,
            discriminator_mode='seen_unseen',
            symmetric=args.symmetric,
            no_texture=args.no_texture,
            num_views=20,
        )
    else:
        model = models.PascalModel(
            encoder_type=args.encoder_type,
            shape_decoder_type=args.shape_decoder_type,
            texture_decoder_type=args.texture_decoder_type,
            discriminator_type=args.discriminator_type,
            silhouette_loss_type=None,
            vertex_scaling=args.vertex_scaling,
            texture_scaling=args.texture_scaling,
            silhouette_loss_levels=0,
            lambda_silhouettes=0,
            lambda_perceptual=0,
            lambda_inflation=0,
            lambda_graph_laplacian=0,
            lambda_discriminator=0,
            no_texture=args.no_texture,
            symmetric=args.symmetric,
            class_conditional=None,
        )
    del model.discriminator
    chainer.serializers.load_npz(os.path.join(directory_model, 'model.npz'), model, strict=False)
    model.to_gpu(args.gpu)
    model.renderer.anti_aliasing = True
    model.renderer.image_size = image_size

    if 'shapenet' in sys.argv:
        cid = None
        for cid2 in class_list_shapenet:
            object_ids = [os.path.basename(d) for d in glob.glob(os.path.join(args.dataset_directory, cid2, '*'))]
            if args.object_id in object_ids:
                cid = cid2
        directory_image = os.path.join(args.dataset_directory, cid, args.object_id)
        filename_image = os.path.join(directory_image, 'render_%d.png' % args.view_id)
        viewpoint = open(os.path.join(directory_image, 'view.txt')).readlines()[args.view_id]
        azimuth, elevation, _, distance = map(float, viewpoint.split())
        azimuth = -azimuth + 90

        image = scipy.misc.imread(filename_image)
        scipy.misc.toimage(image, cmin=0, cmax=255).save('/tmp/%s_%02d.png' % (args.object_id, args.view_id))
        image = image.transpose((2, 0, 1))[None, :, :, :].astype('float32') / 255.
        image = dataset_shapenet.process_images(image)
        image = cp.array(image)
    else:
        dataset = dataset_pascal.Pascal(args.dataset_directory, [args.class_id], 'val')

        image_in = dataset.images_original[args.class_id][args.image_number]
        image_ref = dataset.images_ref[args.class_id][args.image_number].astype('float32') / 255.
        rotation = dataset.rotation_matrices[args.class_id][args.image_number]
        bounding_box = dataset.bounding_boxes[args.class_id][args.image_number]
        image, image_ref, rotation_matrix = dataset_pascal.crop_image(
            image_in,
            image_ref,
            rotation,
            bounding_box,
            padding=0.15,
            jitter=0,
            flip=False,
        )
        scipy.misc.toimage(image, cmin=0, cmax=1).save('/tmp/pascal_%s_%02d.png' % (args.class_id, args.image_number))
        azimuth = -np.degrees(np.arctan(rotation_matrix[0, 2] / rotation_matrix[0, 0]))
        if rotation_matrix[0, 0] < 0:
            azimuth += 180
        elevation = -np.degrees(np.arctan(rotation_matrix[2, 1] / rotation_matrix[1, 1]))
        if rotation_matrix[1, 1] > 0:
            elevation += 180
        print elevation, azimuth
        distance = 2
        image = chainer.cuda.to_gpu(image[None, :, :, :])

    elevation_list = np.array(
        list(np.linspace(elevation, elevation_target, num_rotation_a)) +
        [elevation_target] * num_rotation_b +
        list(np.linspace(elevation_target, elevation, num_rotation_a)))
    azimuth_list = np.array(
        [azimuth] * num_rotation_a +
        list(np.linspace(azimuth, azimuth - 360, num_rotation_b)) +
        [azimuth] * num_rotation_a)
    distance_list = np.array([distance] * (num_rotation_a * 2 + num_rotation_b))
    if 'shapenet' in sys.argv:
        viewpoints = neural_renderer.get_points_from_angles(distance_list, elevation_list, azimuth_list).data
        viewpoints = cp.array(viewpoints).astype('float32')
    else:
        viewpoints = [to_rotation_matrix(e, a) for e, a in zip(elevation_list, azimuth_list)]
        viewpoints = cp.array(viewpoints).astype('float32')

    # print filename_image, viewpoint
    with chainer.using_config('enable_backprop', False):
        with chainer.configuration.using_config('train', False):
            codes = model.encode(image)
            vertices, faces = model.decode_shape(codes)
            if not model.no_texture:
                vertices_t, faces_t, textures = model.decode_texture(codes)

    if 'shapenet' in sys.argv:
        prefix = '%s_%02d_%s' % (args.object_id, args.view_id, args.experiment_id)
    else:
        prefix = 'pascal_%s_%02d_%s' % (args.class_id, args.image_number, args.experiment_id)

    for i, viewpoint in enumerate(viewpoints):
        if model.no_texture:
            if False:
                image_out = model.render(vertices, faces, viewpoint[None, :])
            else:
                vertices = model.rotate(vertices, viewpoint[None, :])

                model.renderer.anti_aliasing = False
                model.renderer.image_size *= 2
                image_out = model.renderer.render_depth(vertices, faces)
                model.renderer.image_size /= 2
                image_out = image_out[0].data.get().squeeze()
                min_v = image_out[image_out != 0].min()
                max_v = image_out[image_out != 0].max()
                image_out[image_out != 0] = ((image_out[image_out != 0] - min_v) / (max_v - min_v)) * 0.8
                image_out[image_out == 0] = 1
                image_out = scipy.misc.imresize(image_out, (image_out.shape[1] / 2, image_out.shape[0] / 2))
        else:
            image_out = model.render(vertices, faces, viewpoint[None, :], vertices_t, faces_t, textures)
            image_out = image_out[0].data.get().transpose((1, 2, 0)).squeeze()

        scipy.misc.toimage(image_out, cmin=0, cmax=1).save('/tmp/%s_%03d.png' % (prefix, i))

    # generate gif (need ImageMagick)
    options = '-delay %f -loop 0 -dispose 2' % (100 / args.frame_rate)
    subprocess.call('convert %s /tmp/%s_*.png /tmp/%s.gif' % (options, prefix, prefix), shell=True)

    # remove temporary files
    # for filename in glob.glob('/tmp/%s_*.png' % prefix):
    #     os.remove(filename)


if __name__ == '__main__':
    run()
