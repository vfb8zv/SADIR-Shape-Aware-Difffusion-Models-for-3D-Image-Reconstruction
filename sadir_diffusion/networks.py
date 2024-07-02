import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from .unet import UNetModel
from .modelio import LoadableModel, store_config_args
from .attention_unet import UNet3D

class SADIR_net(LoadableModel):
 
    @store_config_args
    def __init__(self,
                 inshape,
                 use_attention_unet=True):
        super().__init__()
        self.use_attention_unet = use_attention_unet
        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        
        if self.use_attention_unet==False:
        
            self.unet_model = UNetModel(
                image_size=64,
                in_channels=3,
                model_channels=16,
                out_channels=3,
                num_res_blocks=1,
                attention_resolutions=(16,32),
                channel_mult=(1, 2, 3, 4),
                num_classes= None,
                use_fp16=False,
                num_heads=1,
                num_head_channels=-1,
                num_heads_upsample=-1,
                use_scale_shift_norm=False,
                resblock_updown=False,
                use_new_attention_order=False,
            )
            
        else:
        
            self.unet_model= UNet3D(
                in_channels = 3,
                out_channels = 3,
                f_maps = 32,
                final_sigmoid= False,
                layer_order = 'ceg'
            )
        

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)  #### LAGO NEEDED

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
       
    def forward(self, source):
        # propagate unet
        x = self.unet_model(source)

        # transform into flow field
        if self.use_attention_unet:
            flow_field = x
        else:
            flow_field = self.flow(x)
            
        return flow_field
