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
python PreTraining/mocov2_mgpu_dct_imagenet.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```
