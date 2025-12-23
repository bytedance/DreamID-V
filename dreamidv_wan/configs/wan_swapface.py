# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan T2V 1.3B ------------------------#

swapface = EasyDict(__name__='Config: DreamidV Swapface')
swapface.update(wan_shared_cfg)

# t5
swapface.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
swapface.t5_tokenizer = 'google/umt5-xxl'

# vae
swapface.vae_checkpoint = 'Wan2.1_VAE.pth'
swapface.vae_stride = (4, 8, 8)


# transformer
swapface.model_type = 'i2v'
swapface.patch_size = (1, 2, 2)
swapface.dim = 1536
swapface.ffn_dim = 8960
swapface.freq_dim = 256
swapface.in_dim = 48
swapface.num_heads = 12
swapface.num_layers = 30
swapface.window_size = (-1, -1)
swapface.qk_norm = True
swapface.cross_attn_norm = True
swapface.eps = 1e-6
