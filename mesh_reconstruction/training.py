import numpy as np
import os
import scipy.misc

import chainer

import utils


class Iterator(chainer.dataset.Iterator):
    """
    Iterator that calls self.dataset.get_random_batch at each iteration. Epoch is not used.
    """

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle
        self.epoch = 0
        self.epoch_detail = 0

    def __next__(self):
        return self.dataset.get_random_batch(self.batch_size)

    next = __next__


class Updater(chainer.training.StandardUpdater):
    def __init__(self, model, iterator, optimizers, converter, device=None, loss_func=None, iterative=False):
        super(Updater, self).__init__(iterator, optimizers, converter, device, loss_func)
        self.model = model
        self.optimizers = optimizers
        self.iterative = iterative

        reporter = chainer.Reporter()
        reporter.add_observer('main', model)
        reporter.add_observers('main', model.namedlinks(skipself=True))
        self.reporter = reporter

    def update_core(self):
        if not self.iterative:
            batch = self.converter(self._iterators['main'].next(), self.device)
            loss = self.model(*batch)

            self._optimizers['encoder'].target.cleargrads()
            self._optimizers['shape_decoder'].target.cleargrads()
            self._optimizers['texture_decoder'].target.cleargrads()
            self._optimizers['discriminator'].target.cleargrads()
            loss.backward()
            self._optimizers['encoder'].update()
            self._optimizers['shape_decoder'].update()
            self._optimizers['texture_decoder'].update()
            self._optimizers['discriminator'].update()
            del loss
        else:
            batch = self.converter(self._iterators['main'].next(), self.device)
            loss_generator, loss_discriminator = self.model(*batch)

            self._optimizers['encoder'].target.cleargrads()
            self._optimizers['shape_decoder'].target.cleargrads()
            self._optimizers['texture_decoder'].target.cleargrads()
            loss_generator.backward()
            self._optimizers['encoder'].update()
            self._optimizers['shape_decoder'].update()
            self._optimizers['texture_decoder'].update()
            del loss_generator

            self._optimizers['discriminator'].target.cleargrads()
            loss_discriminator.backward()
            self._optimizers['discriminator'].update()
            del loss_discriminator


def converter(data, device=None):
    """
    Converting to GPU.
    """

    ret = []
    for d in data:
        if d is not None:
            ret.append(chainer.cuda.to_gpu(d, device))
        else:
            ret.append(None)
    return tuple(ret)


def validation(trainer=None, model=None, dataset=None, batch_size=100, num_views=None):
    # evaluate IoU between voxels on all classes
    with chainer.using_config('enable_backprop', False):
        with chainer.configuration.using_config('train', False):
            ious = {}
            for class_id in dataset.class_ids:
                iou = []
                for batch in dataset.get_all_batches_for_evaluation(batch_size, class_id, num_views=num_views):
                    images_in, voxels = converter(batch)
                    iou += model.evaluate_iou(images_in, voxels).tolist()
                iou = sum(iou) / len(iou)
                ious['%s/iou_%s' % (dataset.set_name, class_id)] = iou

            ious['%s/iou' % dataset.set_name] = np.mean([float(v) for v in ious.values()])
            chainer.report(ious)


def draw(trainer=None, model=None, batch=None, prefix=None):
    # save input & predicted images
    with chainer.using_config('enable_backprop', False):
        with chainer.using_config('train', False):
            directory = trainer.out

            if model.dataset_name == 'shapenet':
                images_in, _, viewpoints, _, _ = converter(batch)
                images_prediction = model.predict_and_render(images_in, viewpoints).data.get().squeeze()
                images_prediction_cp = model.predict_and_render(images_in, viewpoints[::-1]).data.get().squeeze()
                images_ref = None
                cmin_in = -1
            elif model.dataset_name == 'pascal':
                images_in, images_ref, rotation_matrices, rotation_matrices_random, labels = converter(batch)
                images_prediction = model.predict_and_render(images_in, rotation_matrices).data.get().squeeze()
                images_prediction_cp = model.predict_and_render(images_in, rotation_matrices_random).data.get().squeeze()
                cmin_in = 0

            # input images
            image = utils.tile_images(images_in.get())
            scipy.misc.toimage(image, cmin=cmin_in, cmax=1).save(os.path.join(directory, '%s_images_in.png' % prefix))
            if images_ref is not None:
                image = utils.tile_images(images_ref.get())
                scipy.misc.toimage(image, cmin=cmin_in, cmax=1).save(os.path.join(directory, '%s_images_ref.png' % prefix))

            # predicted images
            images_prediction = utils.tile_images(images_prediction)
            if model.no_texture:
                images_prediction = 1 - images_prediction
            scipy.misc.toimage(
                images_prediction, cmin=0, cmax=1).save(os.path.join(directory, '%s_prediction.png' % prefix))

            # predicted images from viewpoints
            images_prediction_cp = utils.tile_images(images_prediction_cp)
            if model.no_texture:
                images_prediction_cp = 1 - images_prediction_cp
            scipy.misc.toimage(
                images_prediction_cp, cmin=0, cmax=1).save(os.path.join(directory, '%s_prediction_cp.png' % prefix))


class WeightDecay(object):
    name = 'WeightDecay'
    timing = 'pre'
    call_for_each_param = True

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, rule, param):
        p, g = param.data, param.grad
        if p is None or g is None:
            return
        if param.ndim < 2:
            return
        with chainer.cuda.get_device_from_array(p) as dev:
            if int(dev) == -1:
                g += self.rate * p
            else:
                kernel = chainer.cuda.elementwise(
                    'T p, T decay', 'T g', 'g += decay * p', 'weight_decay')
                kernel(p, self.rate, g)
