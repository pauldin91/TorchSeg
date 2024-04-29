# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict
import torch.utils.model_zoo as model_zoo

C = edict()
base_config = C
cfg = C
C.channels='rgb'
C.dataset='fire'
if C.dataset == 'fire':
    C.dataset=C.channels

if C.channels == 'rgb':
    C.network=3
    C.resnet=3
    C.image_mean = np.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
    C.image_std = np.array([0.229, 0.224, 0.225])
else:
    C.network=6
    C.resnet=6
    C.image_mean = np.array([0.485, 0.456, 0.406,0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
    C.image_std = np.array([0.229, 0.224, 0.225,0.229, 0.224, 0.225])

C.image_size   = 512
C.image_height = 512
C.image_width  = 512