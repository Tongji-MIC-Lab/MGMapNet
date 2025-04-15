import torch
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      POSITIONAL_ENCODING,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, BaseTransformerLayer
from mmcv.cnn.bricks.transformer import TransformerLayerSequence,build_feedforward_network
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import copy
import math
def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTRDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(MapTRDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
#########################
        #TODO point query ffn and norm
        # import ipdb;ipdb.set_trace()
        ffn_cfg={'type': 'FFN', 'embed_dims': 256, 'feedforward_channels': 512, 'num_fcs': 2, 'ffn_drop': 0.1, 'act_cfg': {'type': 'ReLU', 'inplace': True}}
        ffn_layer=build_feedforward_network(ffn_cfg,dict(type='FFN'))
        ffn_norm=nn.LayerNorm(256)
        num_layers=6
        self.point_ffn=_get_clones(ffn_layer,num_layers)
        self.point_norm=_get_clones(ffn_norm,num_layers)
        # pt_pos_query_proj = nn.Sequential(
        #     nn.Linear(20*self.embed_dims, self.embed_dims),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dims, self.embed_dims),
        # )
        # self.pt_pos_query_projs = _get_clones(pt_pos_query_proj, 6)

        #TODO point pos embed
        pos_embed=nn.Sequential(
            nn.Linear(20*2, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.gen_pos_embed = _get_clones(pos_embed, num_layers)

#########################
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
        output_instance = query
        num_query, bs, embed_dims=query.shape
        intermediate = []
        intermediate_instance = []
        intermediate_reference_points = []
        intermediate_MGA_reference_points = []

        output_point=None
        point_pe=None
        point_query_pos=None
        reference_points=reference_points.reshape(bs,num_query,20,2)
        
        for lid, layer in enumerate(self.layers):
            
            reference_points_input=reference_points
            if lid>0:
                reference_points_reshape = reference_points.view(bs, -1, 20 * 2)
                point_query_pos = self.gen_pos_embed[lid](reference_points_reshape)
                point_query_pos = point_query_pos.permute(1, 0, 2)

            output_instance,output_point,point_pe,ref_out = layer(
                output_instance,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                point_query_pre=output_point,
                point_pe_pre=point_pe,
                sin_pos=point_query_pos,
                **kwargs)


            output_point=self.point_ffn[lid](output_point)
            output_point=self.point_norm[lid](output_point) #[4, 50, 20, 256]
            output_point=output_point.reshape(bs,num_query,20,256)
            intermediate.append(output_point)
            if reg_branches is not None:
                tmp = reg_branches[lid](output_point).reshape(bs,num_query,20,2)


                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            # output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate_MGA_reference_points.append(ref_out)
                intermediate_reference_points.append(reference_points)
                intermediate_instance.append(output_instance)
                # intermediate_sampling_location.append(sampling_location)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points),torch.stack(intermediate_instance),torch.stack(intermediate_MGA_reference_points)

        return output, reference_points




@TRANSFORMER_LAYER.register_module()
class DecoupledDetrTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 num_vec=50,
                 num_pts_per_vec=20,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(DecoupledDetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        # assert len(operation_order) == 8
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.
        **kwargs contains some specific arguments of attentions.
        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'
        # 
        num_vec = kwargs['num_vec']
        num_pts_per_vec = kwargs['num_pts_per_vec']
        for layer in self.operation_order:
            if layer == 'self_attn':
            
             
                n_vec, n_batch, n_dim = query.shape
                temp_key = temp_value = query
                query= self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=kwargs['self_attn_mask'],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                query = query.view(num_vec, n_batch, n_dim)
                query_pos = query_pos.view(num_vec, n_batch, n_dim)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query,query_point,query_pe,ref_out  = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        # return query
        return query,query_point,query_pe,ref_out

