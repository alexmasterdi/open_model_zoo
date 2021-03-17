"""
 Copyright (C) 2020-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import cv2
import numpy as np
from skimage.measure import block_reduce

from .hpe_associative_embedding import AssociativeEmbeddingDecoder
from .model import Model
from .utils import resize_image


class HpeHRNet(Model):
    def __init__(self, ie, model_path, target_size, aspect_ratio, prob_threshold, size_divisor=32):
        super().__init__(ie, model_path)
        self.image_blob_name = self._get_inputs(self.net)
        self.heatmaps_blob_name = find_layer_by_name('heatmaps', self.net.outputs)
        self.nms = HeatmapNMS(kernel=5)
        self.embeddings_blob_name = find_layer_by_name('embeddings', self.net.outputs)

        self.output_scale = self.net.input_info[self.image_blob_name].input_data.shape[-1] / self.net.outputs[self.heatmaps_blob_name].shape[-1]

        if target_size is None:
            h, w = self.net.input_info[self.image_blob_name].input_data.shape[-2:]
            target_size = min(h, w)
        if aspect_ratio >= 1.0:  # img width >= height
            input_height, input_width = target_size, round(target_size * aspect_ratio)
        else:
            input_height, input_width = round(target_size / aspect_ratio), target_size
        self.h = (input_height + size_divisor - 1) // size_divisor * size_divisor
        self.w = (input_width + size_divisor - 1) // size_divisor * size_divisor
        default_input_shape = self.net.input_info[self.image_blob_name].input_data.shape
        input_shape = {self.image_blob_name: (default_input_shape[:-2] + [self.h, self.w])}
        self.logger.info('Reshape net to {}'.format(input_shape))
        self.net.reshape(input_shape)

        self.decoder = AssociativeEmbeddingHRNetDecoder(
            num_joints=self.net.outputs[self.heatmaps_blob_name].shape[1],
            adjust=True,
            refine=True,
            delta=0.0,
            max_num_people=30,
            detection_threshold=0.1,
            tag_threshold=1,
            pose_threshold=prob_threshold,
            use_detection_val=True,
            ignore_too_much=False,
            dist_reweight=True)
        self.size_divisor = size_divisor

    @staticmethod
    def _get_inputs(net):
        image_blob_name = None
        for blob_name, blob in net.input_info.items():
            if len(blob.input_data.shape) == 4:
                image_blob_name = blob_name
            else:
                raise RuntimeError('Unsupported {}D input layer "{}". Only 2D and 4D input layers are supported'
                                   .format(len(blob.shape), blob_name))
        if image_blob_name is None:
            raise RuntimeError('Failed to identify the input for the image.')
        return image_blob_name

    def preprocess(self, inputs):
        img = resize_image(inputs, (self.w, self.h), keep_aspect_ratio=True)
        h, w = img.shape[:2]
        if not (self.h - self.size_divisor < h <= self.h and self.w - self.size_divisor < w <= self.w):
            self.logger.warn("Chosen model aspect ratio doesn't match image aspect ratio")
        resize_img_scale = np.array((inputs.shape[1] / w, inputs.shape[0] / h), np.float32)

        pad = [(self.h - h + 1) // 2, (self.w - w + 1) // 2, (self.h - h) // 2, (self.w - w) // 2]
        img = np.pad(img, ((pad[0], pad[2]), (pad[1], pad[3]), (0, 0)),
                     mode='constant', constant_values=0)
        img = img.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        img = img[None]
        return {self.image_blob_name: img}, resize_img_scale

    def postprocess(self, outputs, resize_img_scale):
        heatmaps = outputs[self.heatmaps_blob_name]
        aembds = outputs[self.embeddings_blob_name]
        # resize lr_heatmaps and aembds to heatmaps size
        aembds = cv2.resize(np.transpose(aembds[0], (1, 2, 0)), heatmaps.shape[2:4][::-1])
        aembds = np.transpose(aembds, (2, 0, 1))[None]
        num_joints = heatmaps.shape[1]
        # average of heatmaps and apply nms
        heatmaps = (heatmaps + aembds[:, :num_joints, :, :]) / 2
        nms_heatmaps = self.nms(heatmaps)
        aembds = aembds[:, num_joints:, :, :]
        poses, scores = self.decoder(heatmaps, aembds, nms_heatmaps=nms_heatmaps)
        # Rescale poses to the original image.
        poses[:, :, :2] *= resize_img_scale * self.output_scale
        return poses, scores


def find_layer_by_name(name, layers):
    suitable_layers = [layer_name for layer_name in layers if layer_name.startswith(name)]
    if not suitable_layers:
        raise ValueError('Suitable layer for "{}" output is not found'.format(name))

    if len(suitable_layers) > 1:
        raise ValueError('More than 1 layer matched to "{}" output'.format(name))

    return suitable_layers[0]


class HeatmapNMS:
    def __init__(self, kernel):
        self.kernel = kernel
        self.pad = (kernel - 1) // 2

    def max_pool(self, x):
        # Max pooling kernel x kernel with stride 1 x 1.
        k = self.kernel
        p = self.pad
        pooled = np.zeros_like(x)
        hmap = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        h, w = hmap.shape[-2:]
        for i in range(k):
            n = (h - i) // k * k
            for j in range(k):
                m = (w - j) // k * k
                hmap_slice = hmap[..., i:i + n, j:j + m]
                pooled[..., i::k, j::k] = block_reduce(hmap_slice, (1, 1, k, k), np.max)
        return pooled

    def __call__(self, heatmaps):
        pooled = self.max_pool(heatmaps)
        return heatmaps * (pooled == heatmaps).astype(heatmaps.dtype)


class AssociativeEmbeddingHRNetDecoder(AssociativeEmbeddingDecoder):
    @staticmethod
    def adjust(ans, heatmaps):
        H, W = heatmaps.shape[-2:]
        for n, people in enumerate(ans):
            for person in people:
                for k, joint in enumerate(person):
                    if joint[2] > 0:
                        heatmap = heatmaps[n, k]
                        px = int(joint[0])
                        py = int(joint[1])

                        diff = np.array([
                            heatmap[py, min(px + 1, W - 1)] - heatmap[py, max(px - 1, 0)],
                            heatmap[min(py + 1, H - 1), px] - heatmap[max(py - 1, 0), px]
                        ])
                        joint[:2] += np.sign(diff) * .25 + 0.5
        return ans

    @staticmethod
    def refine(heatmap, tag, keypoints, pose_tag=None):
        K, H, W = heatmap.shape
        if len(tag.shape) == 3:
            tag = tag[..., None]

        if pose_tag is not None:
            prev_tag = pose_tag
        else:
            tags = []
            for i in range(K):
                if keypoints[i, 2] > 0:
                    x, y = keypoints[i][:2].astype(int)
                    tags.append(tag[i, y, x])
            prev_tag = np.mean(tags, axis=0)

        for i, (_heatmap, _tag) in enumerate(zip(heatmap, tag)):
            if keypoints[i, 2] > 0:
                continue
            # Get position with the closest tag value to the pose tag.
            diff = np.abs(_tag[..., 0] - prev_tag) + 0.5
            diff = diff.astype(np.int32).astype(_heatmap.dtype)
            diff -= _heatmap
            idx = diff.argmin()
            y, x = np.divmod(idx, _heatmap.shape[-1])
            # Corresponding keypoint detection score.
            val = _heatmap[y, x]

            if val > 0:
                keypoints[i, :3] = x, y, val
                diff = np.array([
                    _heatmap[y, min(x + 1, W - 1)] - _heatmap[y, max(x - 1, 0)],
                    _heatmap[min(y + 1, H - 1), x] - _heatmap[max(y - 1, 0), x]
                ])
                keypoints[i, :2] += np.sign(diff) * .25 + 0.5

        return keypoints
