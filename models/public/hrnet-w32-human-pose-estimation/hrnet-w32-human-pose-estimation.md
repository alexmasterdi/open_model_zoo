# hrnet-w32-human-pose-estimation

## Use Case and High-Level Description

The `HRNet-W32` model is one of the [HigherHRNet](https://arxiv.org/pdf/1908.10357).
`HigherHRNet` is a novel bottom-up human pose
estimation method for learning scale-aware representations using high-resolution feature pyramids. The network uses HRNet as backbone, followed by one or more deconvolution modules to generate multi-resolution and high-resolution heatmaps. For every person in an image, the network detects a human pose: a body skeleton consisting of keypoints and connections between them. The pose may contain up to 17 keypoints: ears, eyes, nose, shoulders, elbows, wrists, hips, knees, and ankles.
This is PyTorch\* implementation pretrained on COCO dataset.
For details about implementation of model, check out the [HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) repository.

## Specification

| Metric            | Value                  |
|-------------------|------------------------|
| Type              | Human pose estimation  |
| GFLOPs            | 92.8364                |
| MParams           | 28.6180                |
| Source framework  | PyTorch\*              |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Average Precision (AP)  | 65,03%          | 65,03%           |

## Input

### Original Model

Image, name - `image`,  shape - `1,3,512,512`, format is `B,C,H,W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`. Mean values - [123.675,116.28,103.53], scale values - [58.395,57.12,57.375].

### Converted Model

Image, name - `image`,  shape - `1,3,512,512`, format is `B,C,H,W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

The net outputs two blobs:

- `heatmaps` of shape B, 17, 256, 256 containing location heatmaps for keypoints of all types.
- `heatmaps_lr_and_embeddings` of shape B, 34, 128, 128 containing location heatmaps with resolution `128x128`and associative embedding values, which are used for grouping individual keypoints into poses.

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/HRNet/HigherHRNet-Human-Pose-Estimation/master/LICENSE):

```
MIT License

Copyright (c) 2019 HRNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```