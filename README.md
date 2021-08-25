## Axial Attention

[![PyPI version](https://badge.fury.io/py/axial-attention.svg)](https://badge.fury.io/py/axial-attention)

Implementation of <a href="https://arxiv.org/abs/1912.12180">Axial attention</a> in Pytorch. A simple but powerful technique to attend to multi-dimensional data efficiently. It has worked wonders for me and many other researchers.

Simply add some positional encoding to your data and pass it into this handy class, specifying which dimension is considered the embedding, and how many axial dimensions to rotate through. All the permutating, reshaping, will be taken care of for you.

This paper was actually rejected on the basis of being too simple. And yet, it has since been used successfully in a number of applications, among those <a href="https://ai.googleblog.com/2020/03/a-neural-weather-model-for-eight-hour.html">weather prediction</a>, <a href="https://ai.googleblog.com/2020/08/axial-deeplab-long-range-modeling-in.html"> all-attention image segmentation</a>. Just goes to show.

### Install

```bash
$ pip install axial_attention
```

### Usage

Image

```python
import torch
from axial_attention import AxialAttention

img = torch.randn(1, 3, 256, 256)

attn = AxialAttention(
    dim = 3,               # embedding dimension
    dim_index = 1,         # where is the embedding dimension
    dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
    heads = 1,             # number of heads for multi-head attention
    num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
    sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
)

attn(img) # (1, 3, 256, 256)
```

Channel-last image latents

```python
import torch
from axial_attention import AxialAttention

img = torch.randn(1, 20, 20, 512)

attn = AxialAttention(
    dim = 512,           # embedding dimension
    dim_index = -1,      # where is the embedding dimension
    heads = 8,           # number of heads for multi-head attention
    num_dimensions = 2,  # number of axial dimensions (images is 2, video is 3, or more)
)

attn(img) # (1, 20, 20 ,512)
```

Video

```python
import torch
from axial_attention import AxialAttention

video = torch.randn(1, 5, 128, 256, 256)

attn = AxialAttention(
    dim = 128,           # embedding dimension
    dim_index = 2,       # where is the embedding dimension
    heads = 8,           # number of heads for multi-head attention
    num_dimensions = 3,  # number of axial dimensions (images is 2, video is 3, or more)
)

attn(video) # (1, 5, 128, 256, 256)
```

Image Transformer, with reversible network

```python
import torch
from torch import nn
from axial_attention import AxialImageTransformer

conv1x1 = nn.Conv2d(3, 128, 1)

transformer = AxialImageTransformer(
    dim = 128,
    depth = 12,
    reversible = True
)

img = torch.randn(1, 3, 512, 512)

transformer(conv1x1(img)) # (1, 3, 512, 512)
```

With axial positional embedding

```python
import torch
from axial_attention import AxialAttention, AxialPositionalEmbedding

img = torch.randn(1, 512, 20, 20)

attn = AxialAttention(
    dim = 512,
    heads = 8,
    dim_index = 1
)

pos_emb = AxialPositionalEmbedding(
    dim = 512,
    shape = (20, 20)
)

img = pos_emb(img)  # (1, 512, 20, 20)  - now positionally embedded
img = attn(img)     # (1, 512, 20, 20)
```

## Citation

```bibtex
@misc{ho2019axial,
    title  = {Axial Attention in Multidimensional Transformers},
    author = {Jonathan Ho and Nal Kalchbrenner and Dirk Weissenborn and Tim Salimans},
    year   = {2019},
    archivePrefix = {arXiv}
}
```

```bibtex
@misc{wang2020axialdeeplab,
    title   = {Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation},
    author  = {Huiyu Wang and Yukun Zhu and Bradley Green and Hartwig Adam and Alan Yuille and Liang-Chieh Chen},
    year    = {2020},
    eprint  = {2003.07853},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@inproceedings{huang2019ccnet,
    title   = {Ccnet: Criss-cross attention for semantic segmentation},
    author  = {Huang, Zilong and Wang, Xinggang and Huang, Lichao and Huang, Chang and Wei, Yunchao and Liu, Wenyu},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages   = {603--612},
    year    = {2019}
}
```
