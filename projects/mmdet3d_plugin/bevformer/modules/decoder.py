# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import mmcv
import cv2 as cv
import copy
import warnings
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetectionTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(DetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


class MultiheadCrossAttentionLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, 
                 dropout=0.1, activation="relu", 
                 num_heads=8):
        super().__init__()

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,batch_first=True)
        # self.cross_attn = MSDeformAttn(d_model, n_levels, num_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, query, query_pos, key, key_pos=None):
        """forward
        """
        # cross attention
        
        output = self.cross_attn(
                self.with_pos_embed(query, query_pos),
                self.with_pos_embed(key, key_pos),
                key,
            )[0]
        query = query + self.dropout1(output)
        query = self.norm1(query)
        query = self.forward_ffn(query)
        return query
    
class MultiheadSelfAttentionLayer(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """with_pos_embed
        """
        return tensor if pos is None else tensor + pos

    def forward(self, query, query_pos):
        """forward
        """
        v = query
        q = k = self.with_pos_embed(query, query_pos)
        output = self.self_attn(q, k, v)[0]
        query = query + self.dropout(output)
        query = self.norm(query)
        return query
 


def my_multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations, attention_weights):
    """CPU version of multi-scale deformable attention.

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, # NChw [32, 32, 200, 100]
            sampling_grid_l_, #NHW2 [32, 50, 160, 2]
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_) #NCHW [32, 32, 50, 160]
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    # import ipdb;ipdb.set_trace()
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    #attention_weights.shape [32, 1, 50, 160] 
    #torch.stack(sampling_value_list, dim=-2).flatten(-2) [32, 32, 50, 160]
    #(torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1) [32, 32, 50]
    return output.transpose(1, 2).contiguous(),sampling_value_list 
    # 4 50 20 256 point query CA的范围 4 1000 256 / 200 20 256

# global index=0
#sampling offset delta
from mmcv.cnn.bricks.transformer import MultiheadAttention
@ATTENTION.register_module()
class CustomMSDeformableAttention(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                #  num_points=4,
                 num_points=20,
                 sampling_num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 use_sampling_offsets=True,
                #  use_sampling_offsets=False,
                 num_ins=50,
                 num_pos=20,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False
        # self.point_SA=MultiheadAttention(embed_dims, num_heads=num_heads, dropout=dropout,batch_first=True)
        # self.point_norm=nn.LayerNorm(embed_dims)
        # self.instance_point_CA=MultiheadAttention(embed_dims, num_heads=num_heads, dropout=dropout,batch_first=True)
        # self.instance_point_norm=nn.LayerNorm(embed_dims)
        self.p2p_attention=MultiheadCrossAttentionLayer(embed_dims)
        self.point_pos=nn.Sequential(
            nn.Linear(sampling_num_points*2,embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims,embed_dims)
        )
        self.p2i_attention=MultiheadCrossAttentionLayer(embed_dims)
        self.instance_pos=nn.Sequential(
            nn.Linear(sampling_num_points*num_points*2,embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims,embed_dims)
        )



        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_num_points = sampling_num_points
        self.use_sampling_offsets = use_sampling_offsets
        self.num_ins = num_ins
        self.num_pos=num_pos
        if use_sampling_offsets:
            # TODO delta sampling
            # self.sampling_offsets = nn.Linear(   
            #     embed_dims, num_heads * num_levels  * num_points * 2)
            self.sampling_offsets = nn.Linear(
                embed_dims, num_heads * self.sampling_num_points * num_levels  * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * self.sampling_num_points * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj_instance = nn.Linear(embed_dims, embed_dims)
        self.output_proj_point =nn.Linear(embed_dims, embed_dims)
        self.init_weights()
        # self.index=0
        self.point2ins_proj=nn.Sequential(
            nn.Linear(20*embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims)
        )

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        if self.use_sampling_offsets:
            constant_init(self.sampling_offsets, 0.)
            thetas = torch.arange(
                self.num_heads,
                dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init /
                        grid_init.abs().max(-1, keepdim=True)[0]).view(
                self.num_heads, 1, 1, 1, 2).repeat(1, self.num_points,self.num_levels, self.sampling_num_points, 1)
            # grid_init = (grid_init /
            #             grid_init.abs().max(-1, keepdim=True)[0]).view(
            #     1, 1, 1, self.sampling_num_points, 2).repeat(self.num_heads, self.num_points,self.num_levels, 1, 1)
            
            for i in range(self.sampling_num_points):
                grid_init[:, :, :, i, :] *= i + 1
            # for i in range(self.num_heads):
            #     grid_init[i, :, :, :, :] *= i + 1
            # import ipdb;ipdb.set_trace()
            self.sampling_offsets.bias.data = grid_init.view(-1)
        
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj_instance, distribution='uniform', bias=0.)
        xavier_init(self.output_proj_point, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None, #4,50,20,2
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                point_query_pre=None,
                point_pe_pre=None,
                sin_pos=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, num_points, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        # import ipdb;ipdb.set_trace()
        # fig, ax = plt.subplots()
        # plt.rcParams['savefig.dpi']=400
        # plt.ylim(0, 1),plt.xlim(0, 1) 
        # reference_points_show=np.array(reference_points.detach().cpu().squeeze(0))
        # for i,instance in enumerate(reference_points_show):
        #     for each in instance:
        #         plt.plot(each[0],each[1] , "o", color='r',ms=2)
        # plt.savefig('ref_fig_without_sampling_init/reference_'+str(self.index)+'.jpg')
        # plt.savefig('ref_fig_without_sampling_init/reference_0.jpg')
        # self.index+=1

        if point_query_pre is not None:
            identity_point=point_query_pre
        if value is None:
            value = query

        if identity is None:
            identity = query
        if sin_pos is not None:
            query = query + sin_pos
        else:
            query = query + query_pos
        # print(query.shape,value.shape)
        #100,4,512   5000,4,512
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2) #4 100 512
            value = value.permute(1, 0, 2) #4 100 512

        bs, num_query, embed_dims = query.shape
        
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        #spatial_shapes (1,2) 50,100
        
        value = self.value_proj(value) #4 100 512   
        # print(key_padding_mask.shape,spatial_shapes.shape)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        # import ipdb; ipdb.set_trace()
        if self.use_sampling_offsets:
            #TODO delta sampling
            # sampling_offsets = self.sampling_offsets(query).view(
            #     bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
            #TODO point repeat
            sampling_offsets = self.sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points,  self.sampling_num_points, 2)
        else:
            sampling_offsets = query.new_zeros((bs, num_query, self.num_heads, self.num_levels, self.num_points, 2))
        

        # import ipdb;ipdb.set_trace()
        attention_weights = self.attention_weights(query)
        attention_weights_instance = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points * self.sampling_num_points)
            #[4, 50, 8, 1, 20, 8]
        attention_weights_instance = attention_weights_instance.softmax(-1) #4 50 8 1 20
       
        #TODO point query attention_weight
        attention_weights_instance = attention_weights_instance.view(bs,num_query,
                                                    self.num_heads,
                                                    self.num_levels,
                                                    self.num_points * self.sampling_num_points
                                                    )   #[4, 50, 8, 1, 160]

        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1) # changed to (h, w)
        _, _, num_points, _ = reference_points.shape
        # TODO 8 point
        #reference_points[:,:,None,None,:,None,:] [4, 350, 1, 1, 20, 1, 2]
        #sampling_offsets [4, 350, 8, 1, 20, 8, 2]
        #sampling_locations [4, 350, 8, 20, 1, 8, 2]

        # import ipdb;ipdb.set_trace()
        sampling_locations = reference_points[:,:,None,None,:,None,:] + \
            (sampling_offsets # (bs, num_queries, num_heads, num_lvls, num_points, sampling_num_points, 2) 
            / offset_normalizer[ None, None, None, :, None, None, :])
        

        #########################################
        # import ipdb;ipdb.set_trace()
        reference_out=sampling_locations.mean(2).mean(2).mean(3) #head level sampling_num_points
        # reference_out[1, 350, 8, 1, 20, 8, 2] -> [1, 350, 20, 2]
        # reference_points [1, 350, 20, 2]
        #########################################
        sampling_locations = sampling_locations.view(bs, num_query, self.num_heads,self.num_levels,self.num_points*
                                                   self.sampling_num_points,2)
            #attention_weights_instance.shape[4, 50, 8, 1, 160] sampling_locations_instance.shape [4, 50, 8, 1, 160, 2]
        # output_instance = MultiScaleDeformableAttnFunction_fp32.apply(
        #         value, spatial_shapes, level_start_index, sampling_locations_instance.contiguous(),
        #         attention_weights_instance, self.im2col_step)
        output_instance,sampling_value_list = my_multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights_instance)
        

        ##############################################
        # import ipdb;ipdb.set_trace()
        # fig, ax = plt.subplots()
        # plt.rcParams['savefig.dpi']=500
        # plt.ylim(0, 1)
        # plt.xlim(0, 1) 
        # reference_spl_show=np.array(sampling_locations.detach().cpu().squeeze(0).mean(1).squeeze(1))
        # for i,instance in enumerate(reference_spl_show):
        #     for each in instance:
        #         plt.plot(each[0],each[1] , "o", color='b',ms=1)
        # plt.savefig('sampling_location/sampling_location0.jpg')
        # plt.savefig('sampling_location_without_init/sampling_location0.jpg')

        ##############################################
        # import ipdb;ipdb.set_trace()
        attention_weights_point = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels , self.num_points , self.sampling_num_points)
        attention_weights_point=attention_weights_point.softmax(-1) #4 50 8 1 20 8
       
        sampling_value = torch.stack(sampling_value_list, dim=-2).flatten(-2).reshape(
            bs*self.num_heads,embed_dims//self.num_heads,num_query,self.num_levels*self.num_points , self.sampling_num_points
            ) #32 32 50 20 8
        attention_weights_point1 = attention_weights_point.transpose(1, 2).reshape(
                bs * self.num_heads, 1, num_query, self.num_levels * self.num_points , self.sampling_num_points)
        # output_point = (sampling_value * attention_weights_point1).sum(dim=-1).view(bs,embed_dims,num_query,
        #                                       self.num_points).permute(0,2,3,1).contiguous()
        output_point = (sampling_value * attention_weights_point1).sum(dim=-1).view(bs,embed_dims,num_query,
                                              self.num_levels, self.num_points).mean(3).permute(0,2,3,1).contiguous()
        
        
        # output_point=output_point+pe
        # sampling_locations [4, 50, 8, 1, 20, 8, 2]
        # attention_weights_point  [4 50 8 1 20 8]
        
        sampling_locations_point=sampling_locations.reshape(bs, num_query, self.num_heads,self.num_levels,self.num_points,
                                                   self.sampling_num_points,2).mean(2).mean(2)
                                                   #[4, 50, 20, 8, 2]->[4, 50, 20, 16]
        attention_weights_point_pe=attention_weights_point.mean(2).mean(2) #[4, 50, 20, 8]
        
        # point_pe=torch.cat((attention_weights_point_pe,sampling_locations_point),dim=3)
                                        #[4, 50, 20, 32]
        
        point_pe=sampling_locations_point*attention_weights_point_pe.unsqueeze(-1)

        point_pe=point_pe.flatten(3,4) #[2, 350, 20, 8,2] - > [2, 350, 20, 16]
        # output_point = output_point.flatten(1,2)  #4 50 20 256 -> 4 1000 256  
        output_point = output_point.flatten(0,1) #4 50 20 256 -> 200 20 256                                      
        # point_pe = self.point_pos(point_pe).flatten(1,2)
        point_pe = self.point_pos(point_pe).flatten(0,1)
        if point_query_pre==None:
            output_point = self.p2p_attention(output_point,point_pe,output_point,point_pe) #[200, 20, 256] / [4,1000,256]
        else:
            point_query_pre=point_query_pre.flatten(0,1)
            point_pe_pre=point_pe_pre.flatten(0,1)
            # point_query_pre=point_query_pre.flatten(1,2)
            # point_pe_pre=point_pe_pre.flatten(1,2)
            point_query_pre1=torch.cat((point_query_pre,output_point),dim=1) #200 40 256
            point_pe_pre1=torch.cat((point_pe_pre,point_pe),dim=1) #
            output_point = self.p2p_attention(output_point,point_pe,point_query_pre1,point_pe_pre1)
        
        #get point query
        #point instace CA

        sampling_locations_instance=sampling_locations.mean(2).mean(2) #4,50,320
        
        attention_weights_instance_pe=attention_weights_instance.mean(2).mean(2) # 4,50,160
        # instance_pe=torch.cat((sampling_locations_instance,attention_weights_instance_pe),dim=2) # 4,50,480
        instance_pe=sampling_locations_instance*attention_weights_instance_pe.unsqueeze(-1)
        instance_pe=instance_pe.flatten(2,3)
        instance_pe = self.instance_pos(instance_pe)


        output_point=output_point+output_instance.flatten(0,1).unsqueeze(1) #TODO point query + instance query
        output_point=output_point.reshape(bs,num_query,20,256).flatten(1,2) ####p2i all
        point_pe=point_pe.reshape(bs,num_query,20,256).flatten(1,2) ####p2i all
        # output_point=output_point.reshape(bs,num_query,20,256).flatten(0,1) ####p2i intra
        # point_pe=point_pe.reshape(bs,num_query,20,256).flatten(0,1) ####p2i intra
        
        
        output_point = self.p2i_attention(output_point, query_pos=point_pe, key=output_instance, key_pos=instance_pe)
        #TODO point_pe+instance_pe [2, 350, 256]->[2, 350, 1, 256]->[2, 350, 20, 256]->[2, 7000, 256]
        # output_point = self.p2i_attention(output_point, query_pos=point_pe+instance_pe.unsqueeze(2).repeat(1,1,20,1).flatten(1,2), key=output_instance, key_pos=instance_pe)
        

        # PE 的实现 point level 
        # 只使用一个坐标或者 8 个坐标+8 个权重 自己学一个坐标
        # 
        # instance  20 个点或者 160 个点 + 160 个权重 480-》256 160 权重 320 个点
        #  （20 * 8 * 2）  *  （20 * 8 * 1） -》 20 * 1 * 2 
        #  
        # output_instance = self.output_proj_instance(output_instance)
        
        output_instance=self.point2ins_proj(output_point.reshape(bs,num_query,20,256).flatten(2,3)) #point2ins_proj
        output_point=self.output_proj_point(output_point).reshape(bs,num_query,self.num_points,-1)
        point_pe=point_pe.reshape(bs,num_query,self.num_points,-1)
        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output_instance = output_instance.permute(1, 0, 2)
        if point_query_pre is not None:
            return self.dropout(output_instance) + identity ,self.dropout(output_point)+identity_point,point_pe, reference_out
        return self.dropout(output_instance) + identity ,self.dropout(output_point),point_pe,reference_out

