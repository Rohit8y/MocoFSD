# MocoFSD : Momentum contrast in Frequency &amp; Spatial Domain

Since the inception of GPU, deep learning is widely used for many computer vision tasks, but most of these supervised
and self-supervised methods use spatial information of the image to learn features. The spatial domain, gives information about the
pixels in the image, whereas the frequency domain gives information about the rate of change of pixels in the spatial domain. The
high frequency component in the frequency domain is used to extract edges in the image and the low frequency component gives
information about smoothness. A new method is proposed called Momentum contrast in Frequency and Spatial Domain
(MocoFSD), which learns feature representation by combining the frequency and spatial domain information. Features learned by
MocoFSD, outperform its self-supervised and supervised counterparts on two downstream tasks, fine-grained image classification,
and image classification.

<p align="center">
  <img src="https://user-images.githubusercontent.com/Rohit8y/MocoFSD/main/.github/images/mocofsd_refined4_drawio.png" width="300">
</p>

