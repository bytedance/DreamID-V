# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn
from einops import rearrange
from typing import List, Tuple
from typing import Tuple, Union
from torch.nn.modules.utils import _triple
from torch import Tensor
import os
from PIL import Image
import numpy as np

def flatten(
    hid: List[torch.FloatTensor],  # List of (*** c)
) -> Tuple[
    torch.FloatTensor,  # (L c)
    torch.LongTensor,  # (b n)
]:
    assert len(hid) > 0
    shape = torch.stack([torch.tensor(x.shape[:-1], device=hid[0].device) for x in hid])
    hid = torch.cat([x.flatten(0, -2) for x in hid])
    return hid, shape

def unflatten(
    hid: torch.FloatTensor,  # (L c) or (L ... c)
    hid_shape: torch.LongTensor,  # (b n)
) -> List[torch.Tensor]:  # List of (*** c) or (*** ... c)
    hid_len = hid_shape.prod(-1)
    hid = hid.split(hid_len.tolist())
    hid = [x.unflatten(0, s.tolist()) for x, s in zip(hid, hid_shape)]
    return hid

class PatchIn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: Union[int, Tuple[int, int, int]],
        dim: int,
    ):
        super().__init__()
        t, h, w = _triple(patch_size)
        self.patch_size = t, h, w
        self.proj = nn.Linear(in_channels * t * h * w, dim)

    def initialize_weights(self):
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)

    def forward(
        self,
        vid: torch.Tensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        if t > 1:
            assert vid.size(2) % t == 1
            vid = torch.cat([vid[:, :, :1]] * (t - 1) + [vid], dim=2)
        vid = rearrange(vid, "b c (T t) (H h) (W w) -> b T H W (t h w c)", t=t, h=h, w=w)
        vid = self.proj(vid)
        return vid


class NaPatchIn(PatchIn):
    def forward(
        self,
        vid: torch.Tensor,  # l c
        vid_shape: torch.LongTensor,

        hybrid_parallel: bool = False,
    ) -> torch.Tensor:
       
        t, h, w = self.patch_size
        if not (t == h == w == 1):
            vid = unflatten(vid, vid_shape)
            for i in range(len(vid)):
                if t > 1 and vid_shape[i, 0] % t != 0:
                    vid[i] = torch.cat([vid[i][:1]] * (t - vid[i].size(0) % t) + [vid[i]], dim=0)
                vid[i] = rearrange(vid[i], "(T t) (H h) (W w) c -> T H W (t h w c)", t=t, h=h, w=w)
            vid, vid_shape = flatten(vid)

        # slice vid after patching in when using sequence parallelism
        # if hybrid_parallel is False:
        #     vid = slice_inputs(vid, dim=0)
        vid = self.proj(vid)
        return vid, vid_shape

class Projector(nn.Module):
    def __init__(self, cin=14, cout=512, vid_dim=1536, patch_size = [1,2,2] ):
        super(Projector, self).__init__()
        self.time_down = nn.Sequential(
            nn.Conv1d(in_channels=cin, out_channels=cin, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels=cin, out_channels=cin, kernel_size=3, stride=2, padding=1)
        )
        self.convs = nn.Sequential(
            nn.Conv2d(cin, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, cout, kernel_size=3,padding=1),
        )
        self.out = NaPatchIn(in_channels=cout, patch_size=patch_size, dim=vid_dim)

        
    def forward(self, x):
        
        x = rearrange(x, "b c f h w -> (b f) c h w")
        h, w = x.shape[-2:]
        x = rearrange(x, 'f c h w -> (h w) c f')
        x = self.time_down(x)
        x = rearrange(x, '(h w) c f -> f c h w', h=h, w=w)

        x = self.convs(x)
   

        h, w = x.shape[-2:]
        x = rearrange(x, 'f c h w ->  f h w c', h=h, w=w)
        x, x_shape = flatten([x])
        
       
        x, x_shape = self.out(x, x_shape)#patchsize上对齐

        return x, x.shape
if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 77, 512, 512)
    projector = Projector(cin=3)
    output_tensor, output_shape = projector(input_tensor)
    print(output_tensor.shape)

