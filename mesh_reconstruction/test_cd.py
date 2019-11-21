import argparse
import numpy as np
import os
import random
import sys

import chainer
import cupy as cp
import tqdm

import dataset_pascal
import dataset_shapenet
import models
import training
import json
import neural_renderer


def sample_points(faces, num_points=16384):
    if faces.ndim == 3:
        faces = faces[None, :, :]
    batch_size = faces.shape[0]
    v1 = faces[:, :, 1] - faces[:, :, 0]
    v2 = faces[:, :, 2] - faces[:, :, 0]
    s = (neural_renderer.cross(v1.reshape((-1, 3)), v2.reshape((-1, 3))).data ** 2).sum(-1) ** 0.5
    s = s.reshape((batch_size, -1))
    s = s / s.sum()
    c = s.cumsum(1)
    p = cp.tile(np.arange(0, 1, 1. / num_points)[None, :], (batch_size, 1))
    i = (p[:, :, None] <= c[:, None, :]).argmax(2)
    vs = cp.zeros((batch_size, num_points, 3), 'float32')
    for bn in range(batch_size):
        v0 = faces[bn, i[bn], 0]
        v1 = faces[bn, i[bn], 1]
        v2 = faces[bn, i[bn], 2]
        r1 = cp.tile(cp.random.uniform(0, 1, v1.shape[0])[:, None], (1, 3))
        r2 = cp.tile(cp.random.uniform(0, 1, v1.shape[0])[:, None], (1, 3))
        v = (v0 * (1 - (r1 ** 0.5)) + v1 * (r1 ** 0.5) * (1 - r2) + v2 * (r1 ** 0.5) * r2)
        vs[bn] = v
    return vs


def run():
    directory_shapenet = '/data/unagi0/kato/datasets/ShapeNetCore.v1'
    directory_rendering = '/home/mil/kato/large_data/lsm/shapenet_release'
    class_list_shapenet = [
        '02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649',
        '03691459', '04090263', '04256520', '04379243', '04401088', '04530566']
    class_list_shapenet = [
        '02958343', '03001627', '03211117', '03636649',
        '03691459', '04090263', '04256520', '04379243', '04401088', '04530566']
    class_list_pascal = ['aeroplane', 'car', 'chair']
    skip_ids = [
        '187f32df6f393d18490ad276cd2af3a4',  # invalid image
        '391fa4da294c70d0a4e97ce1d10a5ae6',  # invalid image
        '50cdaa9e33fc853ecb2a965e75be701c',  # failed to load
    ]

    parser = argparse.ArgumentParser()

    # system
    parser.add_argument('-eid', '--experiment_id', type=str, required=True)
    parser.add_argument('-md', '--model_directory', type=str, default='./data/models')
    parser.add_argument('-dd', '--dataset_directory', type=str, default='./data/dataset')
    parser.add_argument('-rs', '--random_seed', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)

    # components
    parser.add_argument('-vs', '--vertex_scaling', type=float, default=0.01)
    parser.add_argument('-ts', '--texture_scaling', type=float, default=1)

    # training
    parser.add_argument('-bs', '--batch_size', type=int, default=20)
    parser.add_argument('-nt', '--no_texture', type=int, default=1)

    if 'shapenet' in sys.argv:
        # shapenet
        parser.add_argument('-ds', '--dataset', type=str, default='shapnet')
        parser.add_argument('-cls', '--class_ids', type=str, default=','.join(class_list_shapenet))
        parser.add_argument('-sym', '--symmetric', type=int, default=0)

        # components
        parser.add_argument('-et', '--encoder_type', type=str, default='resnet18')
        parser.add_argument('-sdt', '--shape_decoder_type', type=str, default='conv')
        parser.add_argument('-tdt', '--texture_decoder_type', type=str, default='conv')
        parser.add_argument('-dt', '--discriminator_type', type=str, default='shapenet_patch')
    elif 'pascal' in sys.argv:
        # dataset
        parser.add_argument('-ds', '--dataset', type=str, default='pascal')
        parser.add_argument('-cls', '--class_ids', type=str, default=','.join(class_list_pascal))
        parser.add_argument('-sym', '--symmetric', type=int, default=1)

        # components
        parser.add_argument('-et', '--encoder_type', type=str, default='resnet18pt')
        parser.add_argument('-sdt', '--shape_decoder_type', type=str, default='basic_symmetric')
        parser.add_argument('-tdt', '--texture_decoder_type', type=str, default='basic')
        parser.add_argument('-dt', '--discriminator_type', type=str, default='pascal_patch')

    args = parser.parse_args()
    directory_output = os.path.join(args.model_directory, args.experiment_id)
    class_ids = args.class_ids.split(',')

    # set random seed, gpu
    chainer.cuda.get_device(args.gpu).use()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    cp.random.seed(args.random_seed)

    # load dataset
    if args.dataset == 'shapenet':
        dataset = dataset_shapenet.ShapeNet(args.dataset_directory, class_ids, 'test', device=args.gpu)
    else:
        dataset = dataset_pascal.Pascal(args.dataset_directory, class_ids, 'test')

    # setup model & optimizer
    if args.dataset == 'shapenet':
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
            discriminator_mode=None,
            symmetric=args.symmetric,
            no_texture=args.no_texture,
            num_views=20,
        )
    elif args.dataset == 'pascal':
        model = models.PascalModel(
            encoder_type=args.encoder_type,
            shape_decoder_type=args.shape_decoder_type,
            texture_decoder_type=args.texture_decoder_type,
            discriminator_type=args.discriminator_type,
            vertex_scaling=args.vertex_scaling,
            texture_scaling=args.texture_scaling,
            silhouette_loss_levels=args.silhouette_loss_levels,
            lambda_silhouettes=args.lambda_silhouettes,
            lambda_inflation=args.lambda_inflation,
            lambda_discriminator=args.lambda_discriminator,
            lambda_graph_laplacian=args.lambda_graph_laplacian,
            no_texture=args.no_texture,
        )
    del model.discriminator
    chainer.serializers.load_npz(os.path.join(directory_output, 'model.npz'), model, strict=False)
    model.to_gpu(args.gpu)

    with chainer.using_config('enable_backprop', False):
        with chainer.configuration.using_config('train', False):
            for class_id in tqdm.tqdm(dataset.class_ids):

                for batch_num, batch in tqdm.tqdm(
                        enumerate(dataset.get_all_batches_for_evaluation(args.batch_size, class_id))):
                    object_id = dataset.object_ids[class_id][batch_num]
                    filename_ref = '%s/%s/%s/model.obj' % (directory_shapenet, class_id, object_id)

                    v_ref, i_ref = neural_renderer.load_obj(filename_ref, normalization=False)
                    v_ref = chainer.cuda.to_gpu(v_ref)
                    i_ref = chainer.cuda.to_gpu(i_ref)
                    f_ref = v_ref[i_ref]
                    p_ref = sample_points(f_ref).get()[0]

                    images_in, voxels = training.converter(batch)
                    v_p, i_p = model.decode_shape(model.encode(images_in))
                    v_p = v_p.data
                    f_p = v_p[:, i_p]
                    p_p = sample_points(f_p).get()

                    import voxelization
                    v_r = voxelization.voxelize(f_ref[None, :, :] + 0.5, 32)[0].get()
                    v_p = voxelization.voxelize(f_p + 0.5, 32).get()

                    if not os.path.exists('/home/mil/kato/temp/points/%s/%s' % (args.experiment_id, class_id)):
                        os.makedirs('/home/mil/kato/temp/points/%s/%s' % (args.experiment_id, class_id))
                    np.save('/home/mil/kato/temp/points/%s/%s/%s_pr.npy' % (args.experiment_id, class_id, object_id), p_ref)
                    np.save('/home/mil/kato/temp/points/%s/%s/%s_pp.npy' % (args.experiment_id, class_id, object_id), p_p)
                    np.save('/home/mil/kato/temp/points/%s/%s/%s_vr.npy' % (args.experiment_id, class_id, object_id), v_r)
                    np.save('/home/mil/kato/temp/points/%s/%s/%s_vp.npy' % (args.experiment_id, class_id, object_id), v_p)
                    del p_ref, p_p, v_r, v_p, f_p, i_p, images_in, voxels, f_ref, i_ref


if __name__ == '__main__':
    run()
