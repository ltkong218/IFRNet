# IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation
The official PyTorch implementation of [IFRNet](https://arxiv.org/abs/2205.14620) (CVPR 2022).

Authors: [Lingtong Kong](https://scholar.google.com.hk/citations?user=KKzKc_8AAAAJ&hl=zh-CN), [Boyuan Jiang](https://byjiang.com/), Donghao Luo, Wenqing Chu, Xiaoming Huang, [Ying Tai](https://tyshiwo.github.io/), Chengjie Wang, [Jie Yang](http://www.pami.sjtu.edu.cn/jieyang)

Speed, accuracy and parameters comparison on Vimeo90K triplet test dataset.

![](./figures/vimeo90k.png)


## Abstract
Prevailing video frame interpolation algorithms, that generate the intermediate frames from consecutive inputs, typically rely on complex model architectures with heavy parameters or large delay, hindering them from diverse real-time applications. In this work, we devise an efficient encoder-decoder based network, termed IFRNet, for fast intermediate frame synthesizing. It first extracts pyramid features from given inputs, and then refines the bilateral intermediate flow fields together with a powerful intermediate feature until generating the desired output. The gradually refined intermediate feature can not only facilitate intermediate flow estimation, but also compensate for contextual details, making IFRNet do not need additional synthesis or refinement module. To fully release its potential, we further propose a novel task-oriented optical flow distillation loss to focus on learning the useful teacher knowledge towards frame synthesizing. Meanwhile, a new geometry consistency regularization term is imposed on the gradually refined intermediate features to keep better structure layout. Experiments on various benchmarks demonstrate the excellent performance and fast inference speed of proposed approaches. Code is available at [https://github.com/ltkong218/IFRNet](https://github.com/ltkong218/IFRNet).

## Training on Vimeo90K Triplet Dataset for 2x Frame Interpolation
1. Generate optical flow pseudo label 
<pre><code>$ python generate_flow.py</code></pre>

2. Start training
<pre><code>$ python -m torch.distributed.launch --nproc_per_node=4 train_vimeo90k.py --world_size 4 --model_name 'IFRNet' --epochs 300 --batch_size 6 --lr_start 1e-4 --lr_end 1e-5</code></pre>
<pre><code>$ python -m torch.distributed.launch --nproc_per_node=4 train_vimeo90k.py --world_size 4 --model_name 'IFRNet_L' --epochs 300 --batch_size 6 --lr_start 1e-4 --lr_end 1e-5</code></pre>
<pre><code>$ python -m torch.distributed.launch --nproc_per_node=4 train_vimeo90k.py --world_size 4 --model_name 'IFRNet_S' --epochs 300 --batch_size 6 --lr_start 1e-4 --lr_end 1e-5</code></pre>

## Quantitative Comparison
Proposed IFRNet achieves state-of-the-art frame interpolation accuracy with less inference time and computation complexity. We expect proposed single encoder-decoder joint refinement based IFRNet to be a useful component for many frame rate up-conversion and intermediate view synthesis systems.

![](./figures/benchmarks.png)


## Qualitative Comparison
Video comparison for 2x interpolation of methods using 2 input frames on SNU-FILM dataset.

![](./figures/fig2_1.gif)

![](./figures/fig2_2.gif)


## Middlebury Benchmark
Results on the [Middlebury](https://vision.middlebury.edu/flow/eval/results/results-i1.php) online benchmark.

![](./figures/middlebury.png)


## Multi-Frame Interpolation
Quantitative comparison for 8x interpolation of methods using 2 input frames.

<img src=./figures/8x_interpolation.png width=480 />

Qualitative results of IFRNet for 8x interpolation on GoPro and Adobe240 test datasets. Each video has 9 frames where the first and the last frames are input, and the middle 7 frames are predicted by IFRNet.

<p float="left">
  <img src=./figures/fig1_1.gif width=270 />
  <img src=./figures/fig1_2.gif width=270 />
  <img src=./figures/fig1_3.gif width=270 /> 
</p>
