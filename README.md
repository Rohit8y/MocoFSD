# Momentum contrast in frequency &amp; spatial domain <br> for fine-grained image classification

### [**Contents**](#)
1. [Description](#descr)
2. [Installation](#install)
3. [Data Preparation](#prepare)
4. [Self-Supervised Training](#pretrain)
5. [Fine-tuning](#finetune)
6. [References](#ref)

---

### [**Description**](#) <a name="descr"></a>

Momentum contrast in Frequency and Spatial Domain (MocoFSD) inspired by the Moco [[1]](#1) framework learns feature representation by combining the frequency and spatial domain information during the pre-training phase. Features learned by MocoFSD, outperform its self-supervised and supervised counterparts on two downstream tasks, fine-grained image classification,
and image classification.

![mocofsd_refined4_drawio](https://user-images.githubusercontent.com/38680205/193492898-cc243b49-1e82-4e8c-9203-c5a3b471e849.png)

This project was part of the research internship done under [Prof. Jiapan Guo](http://jiapan.nl/) at the [University of Groningen](https://rug.nl) for the course WMCS021-15.

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

### [***Self-supervised Training***](#) <a name="pretrain"></a>

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do self-supervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python pretrain.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

***Note***: for 4-gpu training, we recommend following the [linear lr scaling recipe](https://arxiv.org/abs/1706.02677): `--lr 0.015 --batch-size 128` with 4 gpus.

```
positional arguments:
  DIR                   path to dataset (default: imagenet)

optional arguments:
  --help                show this help message and exit
  --arch                model architecture: resnet18 | resnet34 | resnet50 (default: resnet18)
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

---

### [***Fine-Tuning***](#) <a name="finetune"></a>
Using the pre-trained model we have given the option to fine-tune on four downstream datasets: Stanford Cars, Stanford Dogs, FGVC Aircraft, and DTD. To optimise these models for the downstream task, run:

```
python main.py --arch resnet50 \
--dataset stanfordCars \
--epochs 100
--model <path to checkpoint>
```
We used a grid search to find the optimal value of other hyperparameters.  Once the training process is completed, the final model will be saved by the name <dataset_name>_best_model.pth.tar

---

### [***References***](#) <a name="ref"></a>

<a id="1">[1]</a> 
He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. *In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 9729-9738).*

---
