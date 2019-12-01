# Learning View Priors for Single-view 3D Reconstruction (CVPR 2019)

This is code for a paper [Learning View Priors for Single-view 3D Reconstruction](http://hiroharu-kato.com/projects_en/view_prior_learning.html) by Hiroharu Kato and Tatsuya Harada.

![](http://hiroharu-kato.com/assets/img/view_prior_learning/thumbnail_en.png)

For more details, please visit [project page](http://hiroharu-kato.com/projects_en/view_prior_learning.html).

## Environment
- This code is tested on Python 2.7.

## Testing pretrained models

Download datasets and pretrained models from [here](https://drive.google.com/open?id=1E_2BQbhvFDBeRk5TruMYhPFc0-aVwPhz) and extract them under `data` directory. This can be done by the following commands.
```shell script
mkdir data
cd data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1G5gelwQGniwGgyG92ls_dfc1VtLUiM3s' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1G5gelwQGniwGgyG92ls_dfc1VtLUiM3s" -O dataset.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=119D78nZ329J90yTkfSrq4imRuQ8ON5N_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=119D78nZ329J90yTkfSrq4imRuQ8ON5N_" -O models.zip && rm -rf /tmp/cookies.txt
unzip dataset.zip
unzip models.zip
cd ../
```

Quantitative evaluation of our best model on ShapeNet dataset is done by the following command.

```shell script
python ./mesh_reconstruction/test.py -ds shapenet -nt 0 -eid shapenet_multi_color_nv20_uvr_cc_long
```

This outputs
```shell script
02691156 0.691549002544
02828884 0.59788288686
02933112 0.720973934558
02958343 0.804359183654
03001627 0.603543199669
03211117 0.593105481352
03636649 0.502730883482
03691459 0.673864365473
04090263 0.664089877796
04256520 0.654773500288
04379243 0.602735843742
04401088 0.767574659204
04530566 0.616663414002
all 0.653372787125
```

Other ShapeNet models are listed in `test_shapenet.sh`.

Drawing animated gif of ShapeNet reconstruction requires the dataset provided by [Kar et al. NIPS 2017].

```shell script
cd data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=17GjULuQZsn-s92PQFQSBzezDkonowIxR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17GjULuQZsn-s92PQFQSBzezDkonowIxR" -O lsm.tar.gz && rm -rf /tmp/cookies.txt
tar xvzf lsm.tar.gz
cd shapenet_release/renders/
find ./ -name "*.tar.gz" -exec tar xvzf {} \;
cd ../../../
```

Then, the following commands
```shell script
mkdir tmp
bash make_gif.sh
```

output the following images.

![](https://raw.githubusercontent.com/hiroharu-kato/view_prior_learning/master/data/examples/460cf3a75d8467d1bb579d1d8d989550_00.png)
![](https://raw.githubusercontent.com/hiroharu-kato/view_prior_learning/master/data/examples/460cf3a75d8467d1bb579d1d8d989550_00_shapenet_multi_color_nv20_uvr_cc.gif)

![](https://raw.githubusercontent.com/hiroharu-kato/view_prior_learning/master/data/examples/b4715a33fe6d1bb7f63ee8a34069b7c5_00.png)
![](https://raw.githubusercontent.com/hiroharu-kato/view_prior_learning/master/data/examples/b4715a33fe6d1bb7f63ee8a34069b7c5_00_shapenet_multi_color_nv20_uvr_cc.gif)

![](https://raw.githubusercontent.com/hiroharu-kato/view_prior_learning/master/data/examples/4231883e92a3c1a21c62d11641ffbd35_00.png)
![](https://raw.githubusercontent.com/hiroharu-kato/view_prior_learning/master/data/examples/4231883e92a3c1a21c62d11641ffbd35_00_shapenet_multi_color_nv20_uvr_cc.gif)

## Training

Training requires pre-trained AlexNet model.

```shell script
cd data
mkdir caffemodel
cd caffemodel
wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
cd ../../
```

Training of the provided pre-trained ShapeNet models is done by

```shell script
bash train_shapenet.sh
```

## Citation

```
@InProceedings{kato2019vpl,
    title={Learning View Priors for Single-view 3D Reconstruction},
    author={Hiroharu Kato and Tatsuya Harada},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2019}
}
```
