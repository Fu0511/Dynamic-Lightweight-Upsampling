"""
PSTR: https://arxiv.org/abs/2204.03340
Part Attention layer and its associated decoder.
"""
import math
import warnings

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from mmengine.model import ModuleList
from mmengine.model import BaseModule, constant_init, xavier_init
from mmdet.utils import ConfigType, OptConfigType


class PartAttention(BaseModule):
    """The attention module used in PSTR.

    PSTR: End-to-End One-Step Person Search With Transformers:
    https://arxiv.org/pdf/2010.04159.pdf.

    Args:
        embed_dims (int): The embedding dimension. Default: 256
        num_heads (int): Parallel attention heads. Default: 4
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Default: None.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        im2col_step: int =64,
        dropout: float =0.1,
        batch_first: bool =False,
        norm_cfg: OptConfigType = None,
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg)

        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )

        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
                )
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.normalize_fact = float(embed_dims / num_heads) ** -0.5
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.sampling_offsets, distribution='uniform')

        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)

        xavier_init(self.value_proj, distribution="uniform", bias=0)

        self._is_init = True

    def _pre_attention_forward(
        self,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
    ) -> tuple[int, int, Tensor]:
        """
        Operation to create features to input into self attention.

        Args:
            query (Tensor): Query features from the detector transformer.
                (num_query, bs, embed_dims)
            value (Tensor): Features from a (multiple?) features map from the
                backbone. (num_value, bs, embed_dims)
            reference_points (Tensor): Reference points from the detector
                deformable attention.
            spatial_shapes (Tensor): Height and weight of the features map(s ?)

        Raises:
            ValueError: If spatial shapes matches with num_value

        Returns:
            Tuple[int, int, Tensor, Tensor]: (num_queries, num_heads
                reid_tokens, output) where num_queries and num_values is the
                number of queries and values, output is the features from
                features map(s ?) which position is learned ; there are
                self.num_points of those and reid_tokens is the average
                of output's self.num_points features.
                is learned to be relevant for the ReID task.
        """
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value).view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        bs, _, num_heads, embed_dims = value.shape
        _, num_queries, num_heads, _, _, _ = sampling_locations.shape

        value_list = value.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for level, (H_, W_) in enumerate(spatial_shapes):
            # bs, H_*W_, num_heads, embed_dims ->
            # bs, H_*W_, num_heads*embed_dims ->
            # bs, num_heads*embed_dims, H_*W_ ->
            # bs*num_heads, embed_dims, H_, W_
            value_l_ = (
                value_list[level]
                .flatten(2)
                .transpose(1, 2)
                .reshape(bs * num_heads, embed_dims, H_, W_)
            )
            # bs, num_queries, num_heads, num_points, 2 ->
            # bs, num_heads, num_queries, num_points, 2 ->
            # bs*num_heads, num_queries, num_points, 2
            sampling_grid_l_ = (
                sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
            )
            # bs*num_heads, embed_dims, num_queries, num_points
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampling_value_list.append(sampling_value_l_)
        # (bs, num_queries, num_heads, num_levels, num_points) ->
        output = torch.stack(sampling_value_list, dim=-2).flatten(-2)

        output = output.permute(0, 2, 3, 1)
        reid_tokens = torch.mean(output, dim=2, keepdim=True)
        output = torch.cat([reid_tokens, output], dim=2)

        return num_queries, num_heads, output

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
    ) -> Tensor:
        """Forward Function of PartAttention.

        NOTE (Re-implementation) I guess initial args where copy paster from
        attention forward function. Although, I only kept used args.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        assert self.batch_first, (
            "Deformable DETR forces batch first," "so same for part attention"
        )

        num_queries, num_heads, output = self._pre_attention_forward(
            query,
            value,
            reference_points,
            spatial_shapes,
        )
        batch_size = query.shape[0]

        # 32 = 256 // 8, with default value.
        assert self.embed_dims % num_heads == 0
        head_embed_dims = self.embed_dims // num_heads

        weights = torch.einsum(
            "bqnc,bqcl->bqnl", output * self.normalize_fact, output.transpose(2, 3)
        )
        weights = F.softmax(weights, dim=-1)

        output1 = torch.einsum("bqnl,bqlc->bqnc", weights, output)
        output1 = output1[:, :, 0, :]

        # Concat header features
        output1 = output1.contiguous().view(
            batch_size, num_queries, num_heads * head_embed_dims
        )

        return output1


class PartAttentionDecoderLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        part_attn_cfg (:obj:`ConfigDict` or dict): Config for part
            attention.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(
        self,
        part_attn_cfg: ConfigType = dict(
            num_levels=1,
            num_points=4,
            embed_dims=256,
            batch_first=True,
        ),
        init_cfg: OptConfigType = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.part_attn_cfg = part_attn_cfg

        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.part_attn1 = PartAttention(**self.part_attn_cfg)
        self.part_attn2 = PartAttention(**self.part_attn_cfg)
        self.embed_dims = self.part_attn1.embed_dims

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
    ) -> Tensor:
        """One layer Decoder of a PartAttention decoder from PSTR.

        NOTE (Re-implementation) I guess initial args where copy paster from
        attention forward function. Although, I only kept used args.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        part_attention_intermediate = self.part_attn1(
            query,
            value,
            reference_points,
            spatial_shapes,
        )

        part_attention = self.part_attn2(
            part_attention_intermediate,
            value,
            reference_points,
            spatial_shapes,
        )

        return part_attention


class PartAttentionDecoder(BaseModule):
    """Decoder of PSTR ReID.

    This decoder is similar to the DetrTransformerDecoder but
    FFNs and normalization layers are removed.

    Args:
        num_layers (int): Number of decoder layers.
        layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
            layer. All the layers will share the same config.
        post_norm_cfg (:obj:`ConfigDict` or dict, optional): Config of the
            post normalization layer. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(
        self, num_layers: int, part_attn_cfg: ConfigType, init_cfg: OptConfigType = None
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.part_attn_cfg = part_attn_cfg
        self.num_layers = num_layers
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList(
            [
                PartAttentionDecoderLayer(self.part_attn_cfg)
                for _ in range(self.num_layers)
            ]
        )
        self.embed_dims = self.layers[0].embed_dims

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
    ) -> Tensor:
        """Forward Function of PartAttention.

        NOTE (Re-implementation) I guess initial args where copy paster from
        attention forward function. Although, I only kept used args.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        for layer in self.layers:
            # Adjust reference points with valid ratios
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * valid_ratios[:, None]
                )

            query = layer(
                query,
                value,
                reference_points_input,
                spatial_shapes,
            )

        return query
