import argparse
import functools
import numpy as np
import os
import random
import sys

import chainer
import cupy as cp
import neural_renderer

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
    parser.add_argument('-li', '--log_interval', type=int, default=10000)
    parser.add_argument('-sm', '--save_model', type=int, default=0)
    parser.add_argument('-rs', '--random_seed', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)

    # dataset
    parser.add_argument('-is', '--image_size', type=int, default=224)

    # loss function
    parser.add_argument('-sll', '--silhouette_loss_levels', type=int, default=5)
    parser.add_argument('-ls', '--lambda_silhouettes', type=float, default=1)
    parser.add_argument('-lt', '--lambda_textures', type=float, default=0)
    parser.add_argument('-lp', '--lambda_perceptual', type=float, default=0)
    parser.add_argument('-ld', '--lambda_discriminator', type=float, default=0)
    parser.add_argument('-ld2', '--lambda_discriminator2', type=float, default=0)
    parser.add_argument('-cc', '--class_conditional', type=int, default=0)

    # components
    parser.add_argument('-ts', '--texture_scaling', type=float, default=1)

    # training
    parser.add_argument('-dm', '--discriminator_mode', type=str, default='seen_unseen')
    parser.add_argument('-io', '--iterative_optimization', type=int, default=0)
    parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.5)
    parser.add_argument('-ab2', '--adam_beta2', type=float, default=0.999)

    if 'shapenet' in sys.argv:
        # shapenet
        parser.add_argument('-ds', '--dataset', type=str, default='shapnet')
        parser.add_argument('-cls', '--class_ids', type=str, default=','.join(class_list_shapenet))
        parser.add_argument('-nv', '--num_views', type=int, default=20)
        parser.add_argument('-svt', '--single_view_training', type=int, default=0)
        parser.add_argument('-wov', '--without_viewpoints', type=int, default=0)
        parser.add_argument('-sym', '--symmetric', type=int, default=0)

        # loss function
        parser.add_argument('-slt', '--silhouette_loss_type', type=str, default='l2')
        parser.add_argument('-linf', '--lambda_inflation', type=float, default=1e-4)
        parser.add_argument('-lgl', '--lambda_graph_laplacian', type=float, default=0)
        parser.add_argument('-lel', '--lambda_edge_length', type=float, default=0)

        # training
        parser.add_argument('-bs', '--batch_size', type=int, default=64)
        parser.add_argument('-lr', '--learning_rate', type=float, default=4e-4)
        parser.add_argument('-ni', '--num_iterations', type=int, default=1000000)

        # components
        parser.add_argument('-et', '--encoder_type', type=str, default='resnet18')
        parser.add_argument('-sdt', '--shape_decoder_type', type=str, default='conv')
        parser.add_argument('-tdt', '--texture_decoder_type', type=str, default='conv')
        parser.add_argument('-dt', '--discriminator_type', type=str, default='shapenet_patch')
        parser.add_argument('-vs', '--vertex_scaling', type=float, default=0.01)

    elif 'pascal' in sys.argv:
        # dataset
        parser.add_argument('-ds', '--dataset', type=str, default='pascal')
        parser.add_argument('-cls', '--class_ids', type=str, default=','.join(class_list_pascal))
        parser.add_argument('-sym', '--symmetric', type=int, default=1)

        # loss function
        parser.add_argument('-slt', '--silhouette_loss_type', type=str, default='iou')
        parser.add_argument('-linf', '--lambda_inflation', type=float, default=0)
        parser.add_argument('-lgl', '--lambda_graph_laplacian', type=float, default=0)

        # training
        parser.add_argument('-bs', '--batch_size', type=int, default=16)
        parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5)
        parser.add_argument('-ni', '--num_iterations', type=int, default=1000000)

        # components
        parser.add_argument('-et', '--encoder_type', type=str, default='resnet18pt')
        parser.add_argument('-sdt', '--shape_decoder_type', type=str, default='fc')
        parser.add_argument('-tdt', '--texture_decoder_type', type=str, default='conv')
        parser.add_argument('-dt', '--discriminator_type', type=str, default='pascal_patch')
        parser.add_argument('-vs', '--vertex_scaling', type=float, default=0.1)

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
        dataset_train = dataset_shapenet.ShapeNet(
            args.dataset_directory, class_ids, 'train', args.num_views, args.single_view_training, device=args.gpu)
        dataset_val = dataset_shapenet.ShapeNet(args.dataset_directory, class_ids, 'val', device=args.gpu)
    else:
        dataset_train = dataset_pascal.Pascal(args.dataset_directory, class_ids, 'train')
        dataset_val = dataset_pascal.Pascal(args.dataset_directory, class_ids, 'val')
    iterator_train = training.Iterator(dataset_train, args.batch_size)
    draw_batch_train = dataset_train.get_random_batch(16)
    draw_batch_val = dataset_val.get_random_batch(16)

    no_texture = (args.lambda_textures == 0) and (args.lambda_perceptual == 0)
    if not no_texture:
        import perceptual_loss
        perceptual_loss.get_alex_net()

    # setup model & optimizer
    if args.dataset == 'shapenet':
        if not args.without_viewpoints:
            Model = models.ShapeNetModel
        else:
            Model = models.ShapeNetModelWithoutViewpoint
        model = Model(
            encoder_type=args.encoder_type,
            shape_decoder_type=args.shape_decoder_type,
            texture_decoder_type=args.texture_decoder_type,
            discriminator_type=args.discriminator_type,
            vertex_scaling=args.vertex_scaling,
            texture_scaling=args.texture_scaling,
            silhouette_loss_type=args.silhouette_loss_type,
            silhouette_loss_levels=args.silhouette_loss_levels,
            lambda_silhouettes=args.lambda_silhouettes,
            lambda_textures=args.lambda_textures,
            lambda_perceptual=args.lambda_perceptual,
            lambda_inflation=args.lambda_inflation,
            lambda_discriminator=args.lambda_discriminator,
            lambda_graph_laplacian=args.lambda_graph_laplacian,
            lambda_edge_length=args.lambda_edge_length,
            single_view_training=args.single_view_training,
            class_conditional=args.class_conditional,
            iterative_optimization=args.iterative_optimization,
            discriminator_mode=args.discriminator_mode,
            no_texture=no_texture,
            symmetric=args.symmetric,
            num_views=args.num_views,
        )
        num_views_for_validation = 1
    elif args.dataset == 'pascal':
        model = models.PascalModel(
            encoder_type=args.encoder_type,
            shape_decoder_type=args.shape_decoder_type,
            texture_decoder_type=args.texture_decoder_type,
            discriminator_type=args.discriminator_type,
            silhouette_loss_type=args.silhouette_loss_type,
            vertex_scaling=args.vertex_scaling,
            texture_scaling=args.texture_scaling,
            silhouette_loss_levels=args.silhouette_loss_levels,
            lambda_silhouettes=args.lambda_silhouettes,
            lambda_perceptual=args.lambda_perceptual,
            lambda_inflation=args.lambda_inflation,
            lambda_graph_laplacian=args.lambda_graph_laplacian,
            lambda_discriminator=args.lambda_discriminator,
            no_texture=no_texture,
            symmetric=args.symmetric,
            class_conditional=args.class_conditional,
        )
        num_views_for_validation = None
    model.to_gpu(args.gpu)
    adam_params = {
        'alpha': args.learning_rate,
        'beta1': args.adam_beta1,
        'beta2': args.adam_beta2,
    }
    optimizers = {
        'encoder': neural_renderer.Adam(**adam_params),
        'shape_decoder': neural_renderer.Adam(**adam_params),
        'texture_decoder': neural_renderer.Adam(**adam_params),
        'discriminator': neural_renderer.Adam(**adam_params),
    }
    optimizers['encoder'].setup(model.encoder)
    optimizers['shape_decoder'].setup(model.shape_decoder)
    optimizers['texture_decoder'].setup(model.texture_decoder)
    optimizers['discriminator'].setup(model.discriminator)

    # setup trainer
    updater = training.Updater(
        model, iterator_train, optimizers, converter=training.converter, iterative=args.iterative_optimization)
    trainer = chainer.training.Trainer(updater, stop_trigger=(args.num_iterations, 'iteration'), out=directory_output)
    model.trainer = trainer
    trainer.extend(chainer.training.extensions.LogReport(trigger=(args.log_interval, 'iteration')))
    trainer.extend(
        chainer.training.extensions.PrintReport(
            ['iteration', 'main/loss_silhouettes', 'main/loss_discriminator', 'val/iou', 'elapsed_time']))
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10))
    trainer.extend(
        functools.partial(training.validation, model=model, dataset=dataset_val, num_views=num_views_for_validation),
        name='validation',
        priority=chainer.training.PRIORITY_WRITER,
        trigger=(args.log_interval, 'iteration'))
    trainer.extend(
        functools.partial(training.draw, model=model, batch=draw_batch_val, prefix='val'), name='draw_val',
        trigger=(args.log_interval, 'iteration'))
    trainer.extend(
        functools.partial(training.draw, model=model, batch=draw_batch_train, prefix='train'), name='draw_train',
        trigger=(args.log_interval, 'iteration'))
    trainer.reporter.add_observer('main', model)
    trainer.reporter.add_observers('main', model.namedlinks(skipself=True))

    # main loop
    if True:
        trainer.run()
    else:
        from chainer.function_hooks import TimerHook
        hook = TimerHook()
        import cupy
        from cupy import prof

        with cupy.cuda.profile():
            with cupy.prof.time_range('some range in green', color_id=0):
                with hook:
                    trainer.run()
                hook.print_report()
                print hook.total_time()

    # save model
    if args.save_model:
        chainer.serializers.save_npz(os.path.join(directory_output, 'model.npz'), model)


if __name__ == '__main__':
    run()
