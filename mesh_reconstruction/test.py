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


def run():
    class_list_shapenet = [
        '02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649',
        '03691459', '04090263', '04256520', '04379243', '04401088', '04530566']
    class_list_pascal = ['aeroplane', 'car', 'chair']

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
    parser.add_argument('-bs', '--batch_size', type=int, default=100)
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
        dataset = dataset_pascal.Pascal(args.dataset_directory, class_ids, 'val')

    # setup model & optimizer
    if args.dataset == 'shapenet':
        model = models.ShapeNetModel(
            encoder_type=args.encoder_type,
            shape_decoder_type=args.shape_decoder_type,
            texture_decoder_type=args.texture_decoder_type,
            discriminator_type=args.discriminator_type,
            vertex_scaling=args.vertex_scaling,
            texture_scaling=args.texture_scaling,
            silhouette_loss_type=None,
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
    chainer.serializers.load_npz(os.path.join(directory_output, 'model.npz'), model, strict=False)
    model.to_gpu(args.gpu)

    results = []
    with chainer.using_config('enable_backprop', False):
        with chainer.configuration.using_config('train', False):
            for class_id in tqdm.tqdm(dataset.class_ids):
                iou = []
                for batch in tqdm.tqdm(dataset.get_all_batches_for_evaluation(args.batch_size, class_id)):
                    images_in, voxels = training.converter(batch)
                    iou += model.evaluate_iou(images_in, voxels).tolist()
                iou = sum(iou) / len(iou)
                results.append((class_id, iou))
            results.append(('all', sum([r[1] for r in results]) / len(dataset.class_ids)))

    fp = open(os.path.join(directory_output, 'test.log'), 'w')
    for r in results:
        print r[0], r[1]
        fp.write('%s %f\n' % r)
    fp.close()


if __name__ == '__main__':
    run()
