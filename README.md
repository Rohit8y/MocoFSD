# MocoFSD : Momentum contrast in Frequency &amp; Spatial Domain

### [**Contents**](#)
1. [Description](#descr)
1. [Installation](#install)
2. [Data Preparation](#prepare)

---

### [**Description**](#) <a name="descr"></a>

Momentum contrast in Frequency and Spatial Domain (MocoFSD) inspired by the Moco framework learns feature representation by combining the frequency and spatial domain information during the pre-training phase. Features learned by MocoFSD, outperform its self-supervised and supervised counterparts on two downstream tasks, fine-grained image classification,
and image classification.

![mocofsd_refined4_drawio](https://user-images.githubusercontent.com/38680205/193492898-cc243b49-1e82-4e8c-9203-c5a3b471e849.png)

---

### [**Installation**](#) <a name="install"></a>

**1.** Clone the repository:

``` shell
git clone git@github.com:Rohit8y/MocoFSD.git
cd MocoFSD
```

**2.** Create a new Python environment and activate it:

``` shell
$ python3 -m venv py_env
$ source py_env/bin/activate
```

**3.** Install necessary packages:

``` shell
$ pip install -r requirements.txt
```

---

### [***Data Preparation***](#) <a name="prepare"></a>

- Download the ImageNet dataset from http://www.image-net.org/.
- Then, move and extract the training and validation images to labeled subfolders, using [the following shell script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh)
- The following fine-tuning datasets will be downloaded using [the PyTorch API](https://pytorch.org/vision/stable/datasets.html) automatically in the code.
  - Stanford Dogs
  - Stanford Cars
  - FGVC Aircraft
  - DTD

---

### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python pretrain.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

## Usage

```bash
usage: pretrain.py [-h] [-a ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N] [--lr LR] [--momentum M] [--wd W] [-p N] [--resume PATH] [-e] [--pretrained] [--world-size WORLD_SIZE] [--rank RANK]
               [--dist-url DIST_URL] [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU] [--multiprocessing-distributed] [--dummy]
               [DIR]

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset (default: imagenet)

optional arguments:
  --help            show this help message and exit
  --arch            model architecture: alexnet | convnext_base | convnext_large | convnext_small | convnext_tiny | densenet121 | densenet161 | densenet169 | densenet201 | efficientnet_b0 |
                        efficientnet_b1 | efficientnet_b2 | efficientnet_b3 | efficientnet_b4 | efficientnet_b5 | efficientnet_b6 | efficientnet_b7 | googlenet | inception_v3 | mnasnet0_5 | mnasnet0_75 |
                        mnasnet1_0 | mnasnet1_3 | mobilenet_v2 | mobilenet_v3_large | mobilenet_v3_small | regnet_x_16gf | regnet_x_1_6gf | regnet_x_32gf | regnet_x_3_2gf | regnet_x_400mf | regnet_x_800mf |
                        regnet_x_8gf | regnet_y_128gf | regnet_y_16gf | regnet_y_1_6gf | regnet_y_32gf | regnet_y_3_2gf | regnet_y_400mf | regnet_y_800mf | regnet_y_8gf | resnet101 | resnet152 | resnet18 |
                        resnet34 | resnet50 | resnext101_32x8d | resnext50_32x4d | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | shufflenet_v2_x1_5 | shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn | vit_b_16 | vit_b_32 | vit_l_16 | vit_l_32 | wide_resnet101_2 | wide_resnet50_2 (default: resnet18)
  --workers             number of data loading workers (default: 4)
  --epochs              number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  --batch-size          mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel
  --lr                  initial learning rate
  --momentum            momentum
  --weight-decay        weight decay (default: 1e-4)
  --resume              path to latest checkpoint (default: none)
  --evaluate            evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size          number of nodes for distributed training
  --rank                node rank for distributed training
  --dist-url            url used to set up distributed training
  --dist-backend        distributed backend
  --seed                seed for initializing training.
  --gpu                 GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel
                        training
```
