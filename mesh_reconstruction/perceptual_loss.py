import chainer
import chainer.functions as cf
import chainer.links as cl
import chainer.links.caffe


class AlexNetLoss(chainer.Chain):
    def __init__(self):
        super(AlexNetLoss, self).__init__()
        self.conv1 = cl.Convolution2D(None, 96, 11, stride=4)
        self.conv2 = cl.Convolution2D(None, 256, 5, pad=2)
        self.conv3 = cl.Convolution2D(None, 384, 3, pad=1)
        self.conv4 = cl.Convolution2D(None, 384, 3, pad=1)
        self.conv5 = cl.Convolution2D(None, 256, 3, pad=1)

        # caffe_model = chainer.links.caffe.CaffeFunction('/media/disk2/lab/caffemodel/bvlc_alexnet.caffemodel')
        caffe_model = chainer.links.caffe.CaffeFunction('/home/mil/kato/large_data/caffemodel/bvlc_alexnet.caffemodel')
        self.conv1.W.data = caffe_model.conv1.W.data.copy()
        self.conv1.b.data = caffe_model.conv1.b.data.copy()
        self.conv2.W.data = caffe_model.conv2.W.data.copy()
        self.conv2.b.data = caffe_model.conv2.b.data.copy()
        self.conv3.W.data = caffe_model.conv3.W.data.copy()
        self.conv3.b.data = caffe_model.conv3.b.data.copy()
        self.conv4.W.data = caffe_model.conv4.W.data.copy()
        self.conv4.b.data = caffe_model.conv4.b.data.copy()
        self.conv5.W.data = caffe_model.conv5.W.data.copy()
        self.conv5.b.data = caffe_model.conv5.b.data.copy()
        del caffe_model

    def to_gpu(self, device=None):
        self.conv1.to_gpu(device)
        self.conv2.to_gpu(device)
        self.conv3.to_gpu(device)
        self.conv4.to_gpu(device)
        self.conv5.to_gpu(device)

    def predict_single(self, x):
        # normalization
        xp = chainer.cuda.get_array_module(x)
        mean = xp.array([104, 117, 123], dtype='float32')
        x = x[:, ::-1] * 255 - mean[None, :, None, None]

        h1 = cf.max_pooling_2d(cf.local_response_normalization(cf.relu(self.conv1(x))), 3, stride=2)
        h2 = cf.max_pooling_2d(cf.local_response_normalization(cf.relu(self.conv2(h1))), 3, stride=2)
        h3 = cf.relu(self.conv3(h2))
        h4 = cf.relu(self.conv4(h3))
        h5 = cf.max_pooling_2d(cf.relu(self.conv5(h4)), 3, stride=2)
        return h1, h2, h3, h4, h5

    def __call__(self, x1, x2, eps=1e-5):
        with chainer.using_config('enable_backprop', False):
            hs1 = self.predict_single(x1)
        hs2 = self.predict_single(x2)
        xp = chainer.cuda.get_array_module(x1)
        loss = chainer.Variable(xp.array(0, 'float32'))
        for h1, h2 in zip(hs1, hs2):
            h1 = cf.normalize(h1, axis=1)
            h2 = cf.normalize(h2, axis=1)
            loss += cf.sum(cf.square(h1 - h2)) / (h1.shape[0] * h1.shape[2] * h1.shape[3])
        return loss


alex_net_loss_class = None


def get_alex_net():
    global alex_net_loss_class
    if alex_net_loss_class is None:
        alex_net_loss_class = AlexNetLoss()
        alex_net_loss_class.to_gpu()
    return alex_net_loss_class


def alex_net_features(data):
    alex_net = get_alex_net()
    return alex_net.predict_single(data)


def alex_net_loss(data1, data2):
    alex_net = get_alex_net()
    return alex_net(data1, data2)
