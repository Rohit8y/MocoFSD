# MocoFSD : Momentum contrast in Frequency &amp; Spatial Domain

Momentum contrast in Frequency and Spatial Domain (MocoFSD), which learns feature representation by combining the frequency and spatial domain information. Features learned by MocoFSD, outperform its self-supervised and supervised counterparts on two downstream tasks, fine-grained image classification,
and image classification.

![mocofsd_refined4_drawio](https://user-images.githubusercontent.com/38680205/193492898-cc243b49-1e82-4e8c-9203-c5a3b471e849.png)



### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

This repo aims to be minimal modifications on that code. Check the modifications by:
```
diff PreTraining/mocov2_mgpu_dct_imagenet.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
```

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
