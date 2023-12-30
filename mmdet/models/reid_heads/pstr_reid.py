from collections import defaultdict
from typing import List, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from torch import Tensor
from torch.nn import functional as F

from mmdet.models.dense_heads import (LabeledMatchingLayerQ,
                                      UnlabeledMatchingLayer)
from mmdet.models.layers.transformer import PartAttentionDecoder
from mmdet.models.losses import TripletLoss
from mmdet.registry import MODELS
from mmdet.structures.reid_det_data_sample import (ReIDDetDataSample,
                                                   ReIDDetSampleList)
from mmdet.utils import ConfigType

N_PSTR_DECODER_LAYERS = 3
LOSS_DICT_KEY_TEMPLATE_OIM = "d{}.loss_oim_s{}"
LOSS_DICT_KEY_TEMPLATE_TRIPLET = "d{}.loss_tri_s{}"


# TODO Add docstring
@MODELS.register_module()
class PSTRHeadReID(BaseModel):
    """
    PSTR Head ReID, computes the ReID features from PSTR detections output
    and frame's features maps.

    Args:
        decoder (:obj:`ConfigDict` or dict): Config dict for ReID decoder.
        num_person (int): Number of person IDs in the dataset.
        flag_tri (bool): Compute triplet loss if True, else do not. This is the
            size of the lookup table for loss computation. Default to True.
        queue_size (int): Size of unlabeled queue for loss computation.
            Default to False.

    """

    def __init__(self,
                 decoder: ConfigType,
                 num_person: int,
                 flag_tri: bool = True,
                 queue_size: int = 5000):
        super().__init__()

        self.decoder = PartAttentionDecoder(**decoder)

        self.num_person = num_person
        self.queue_size = queue_size
        self.flag_tri = flag_tri

        self._init_layers()

    def _init_layers(self):
        self.unlabeled_weight = 10
        self.temperature = 15
        self.reid_features_dimension = 256
        self.oim_weight_single_scale_layer = .5
        self.triplet_weight_single_scale_layer = .5

        num_reid_decoder = 3

        self.labeled_matching_layers = nn.ModuleList([
            LabeledMatchingLayerQ(self.num_person,
                                  self.reid_features_dimension)
            for _ in range(num_reid_decoder)
        ])

        self.unlabeled_matching_layers = nn.ModuleList([
            UnlabeledMatchingLayer(self.num_person,
                                   self.reid_features_dimension)
            for _ in range(num_reid_decoder)
        ])

        self.triplet_loss = TripletLoss() if self.flag_tri else None

    def _forward_decoder(
        self,
        detection_decoder_states: Tensor,
        references: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        multi_scale_features_maps_flattened: List[Tensor],
    ) -> Tensor:
        """
        Compute raw ReID features from detector and features maps.

        Args:
            detection_decoder_states (Tensor): The outputs of detector which
                have one in inference and 3 (number of decoder layers) in
                training. (n_decoder_layers, bs, n_queries, n_dim_det)
            references (Tensor): References point from deformable attention.
                (n_decoder_layers, bs, n_queries, 4)
            spatial_shapes (Tensor): The dimension of features maps used in
                detection, before flattening. (n_used_detection, 2)
            valid_ratios (Tensor): Valid ratios of detections, in case some are
                annotatated to be ignored. (bs, ratio_dim, 2).
            multi_scale_features_maps (List[Tensor]): The features maps
                from the backbone/neck. A list of n_scales, each element
                is (bs, , x_dim*y_dim, n_dim_neck).

        Returns:
            Tensor: Result of the raw ReID features
            (n_scales, n_decoder_layers, bs, n_queries, n_dim_reid)
        """
        # Inference ReID: do not input from all 3 layers of the decoder.
        if not self.training:
            assert detection_decoder_states.shape[0] == 1
            last_state = detection_decoder_states

            assert references.shape[0] == 1
            reference = references

            inter_reid_states = [
                self.decoder(
                    query=last_state,
                    value=features_maps_flattened,
                    reference_points=reference,
                    spatial_shapes=spatial_shapes,
                    valid_ratios=valid_ratios,
                ) for features_maps_flattened in
                multi_scale_features_maps_flattened
            ]

            return torch.stack(inter_reid_states).view(
                len(multi_scale_features_maps_flattened), 1, -1)

        number_decoder_layers = references.shape[0]
        assert number_decoder_layers == N_PSTR_DECODER_LAYERS, \
            f"PSTR needs exactly {N_PSTR_DECODER_LAYERS} layers from decoder"

        number_scales = len(multi_scale_features_maps_flattened)
        inter_reid_states = [
            self.decoder(
                query=detection_decoder_states[i_layer],
                value=multi_scale_features_maps_flattened[i_scale],
                reference_points=references[i_layer],
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
            )
            # Per layer of decoder in detector
            for i_layer in range(number_decoder_layers)
            # Per level of scale in backbone/neck
            for i_scale in range(number_scales)
        ]

        inter_reid_states_stacked = torch.stack(inter_reid_states)
        inter_reid_states_reshaped = inter_reid_states_stacked.view(
            number_scales, number_decoder_layers,
            *inter_reid_states_stacked.shape[1:]
            # Delete deformable detr dimension, only use one in Deformable
            # DETR in PSTR.
        )

        return inter_reid_states_reshaped

    def forward(
        self,
        detection_decoder_states: Tensor,
        references: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        multi_scale_features_maps: Tuple[Tensor],
    ) -> list[Tensor]:
        """
        Compute ReID features from detector and features maps.

        Args:
            detection_decoder_states (Tensor): The outputs of detector which
                have one in inference and 3 (number of decoder layers) in
                training. (n_decoder_layers, bs, n_queries, n_dim_det)
            references (Tensor): References point from deformable attention.
                (n_decoder_layers, bs, n_queries, 4)
            spatial_shapes (Tensor): The dimension of features maps used in
                detection, before flattening. (n_used_detection, 2)
            valid_ratios (Tensor): Valid ratios of detections, in case some are
                annotatated to be ignored. (bs, ratio_dim, 2).
            multi_scale_features_maps (Tuple[Tensor]): The features maps
                from the backbone/neck. A n_scales-tuple, each element
                is (bs, n_dim_neck, x_dim, y_dim).

        Returns:
            list[Tensor]: The list of ReID features by scale, n_scale = 3.
                Each scale is (n_decoder_layers, bs, n_queries, n_dim_reid).
        """
        assert detection_decoder_states.shape[0] == references.shape[0]
        assert len(multi_scale_features_maps) == 3

        # Last dimensions (W, H) are flattened
        # Permute spatial dims (num_value) with features dim (n_dim)
        # (num_scales) [(bs, num_value, n_dim)]
        multi_scale_features_maps_flattened = [
            feature_map.flatten(2).permute(0, 2, 1)
            for feature_map in list(multi_scale_features_maps)
        ]

        # (scales, layers_level, batch_size, num_queries, n_features)
        inter_reid_states = self._forward_decoder(
            detection_decoder_states,
            references,
            spatial_shapes,
            valid_ratios,
            multi_scale_features_maps_flattened,
        )

        # [layers_level, ...] (scales)
        reid_outputs: List[Tensor] = [
            inter_reid_state -
            torch.mean(inter_reid_state, dim=2, keepdim=True)
            for inter_reid_state in inter_reid_states
        ]

        return reid_outputs

    def loss_single_layer_scale_sample(
        self,
        reid_features: Tensor,
        assigned_person_ids: Tensor,
        data_sample: ReIDDetDataSample,
        triplet_loss_key: str,
        oim_loss_key: str,
        i_layer: int,
    ) -> dict[str, Tensor]:
        """
        Compute the ReID losses (OIM and Triplet, the former if set).
        For a single decoder layer and a single scale and a single sample.

        Args:
            reid_features (Tensor): ReID features by and sample.
                (n_queries, n_dim_reid)
            assigned_person_ids (Tensor): Result of the
                the matching from assigner, 0 is not assigned and other value
                the person ID assigned to it (-1 is for detection kept but
                no ReID annotations) (n_queries)
            data_sample (ReIDDetDataSample): Annotation of one frame.
            triplet_loss_key (str): key for triplet loss in loss dict. It
                contains information of the layer and the scale which is
                applied.
            oim_loss_key (str): key for oim loss in loss dict. It contains
                information of the layer and the scale which is applied.
            i_layer (int): index of decoder layer.

        Returns:
            dict[str, Tensor]: The keys are the loss based on the input
            (scales) and value is the loss value.
        """
        # No annotations for this frame
        if data_sample.ignored_instances:
            loss_value_zero = reid_features.sum() * 0
            loss_reid = {oim_loss_key: loss_value_zero}
            if self.flag_tri:
                loss_reid[triplet_loss_key] = loss_value_zero
            return loss_reid

        detection_is_assigned = assigned_person_ids != 0
        assigned_reid_features = F.normalize(
            reid_features[detection_is_assigned])
        only_assigned_person_ids = assigned_person_ids[detection_is_assigned]

        labeled_outputs = self.labeled_matching_layers[i_layer](
            assigned_reid_features,
            only_assigned_person_ids,
        )
        (
            labeled_logits,
            labeled_reid_features,
            labeled_person_ids,
        ) = labeled_outputs
        labeled_logits *= self.temperature

        unlabeled_logits = self.unlabeled_matching_layers[i_layer](
            assigned_reid_features, only_assigned_person_ids)
        unlabeled_logits *= self.unlabeled_weight

        matching_scores = torch.cat((labeled_logits, unlabeled_logits), dim=1)

        probabilities = F.softmax(matching_scores, dim=1)
        focal_probabilities = ((1 - probabilities + 1e-12)**2 *
                               (probabilities + 1e-12).log())
        loss_oim = F.nll_loss(
            focal_probabilities,
            only_assigned_person_ids,
            reduction="none",
            ignore_index=-1)

        if not self.flag_tri:
            return {oim_loss_key: loss_oim}

        positive_reid_features = torch.cat(
            (assigned_reid_features, labeled_reid_features))
        positive_person_ids = torch.cat(
            (only_assigned_person_ids, labeled_person_ids))
        loss_triplet = self.triplet_loss(positive_reid_features,
                                         positive_person_ids)

        return {oim_loss_key: loss_oim, triplet_loss_key: loss_triplet}

    def loss_single_layer_scale(
        self,
        batch_reid_features: Tensor,
        batch_assigned_person_ids: Tensor,
        data_samples: ReIDDetSampleList,
        i_layer: int,
        i_scale: int,
    ) -> dict[str, Tensor]:
        """
        Compute the ReID losses (OIM and Triplet, the former if set).
        For a single decoder layer and a single scale.

        Args:
            batch_reid_features (Tensor): ReID features by and sample.
                (bs, n_queries, n_dim_reid)
            batch_assigned_person_ids (Tensor): Result of the
                the matching from assigner, 0 is not assigned and other value
                the person ID assigned to it (-1 is for detection kept but
                no ReID annotations) (bs, n_queries)
            data_samples (ReIDDetSampleList): List of annotations, length
                equals to batch_size.
            i_layer (int): index of decoder layer.
            i_scale (int): index of features map scale.

        Returns:
            dict[str, Tensor]: The keys are the loss based on the input
            (scales) and value is the loss value.
        """
        batch_size = len(data_samples)
        assert batch_size == batch_reid_features.shape[
            0] == batch_assigned_person_ids.shape[0]

        triplet_loss_key = LOSS_DICT_KEY_TEMPLATE_TRIPLET.format(
            i_layer, i_scale)
        oim_loss_key = LOSS_DICT_KEY_TEMPLATE_OIM.format(i_layer, i_scale)

        batch_reid_loss: dict[str, Tensor] = defaultdict()
        for i_sample in range(batch_size):
            sample_reid_loss = self.loss_single_layer_scale_sample(
                batch_reid_features[i_sample],
                batch_assigned_person_ids[i_sample],
                data_samples[i_sample],
                triplet_loss_key,
                oim_loss_key,
                i_layer,
            )

            for sample_loss_key, sample_loss in sample_reid_loss.items():
                if sample_loss_key in batch_reid_loss:
                    batch_reid_loss[sample_loss_key] += (
                        sample_loss * self.oim_weight_single_scale_layer)
                else:
                    batch_reid_loss[sample_loss_key] = (
                        sample_loss * self.triplet_weight_single_scale_layer)

        # Average of losses
        # NOTE: I am not sure that the matching layers losses support reduce
        # method. So this is performed manually
        for loss_key in batch_reid_loss:
            batch_reid_loss[loss_key] /= batch_size

        return batch_reid_loss

    def loss_single_layer(
        self,
        all_scales_batch_reid_features: Tensor,
        batch_assigned_person_ids: Tensor,
        data_samples: ReIDDetSampleList,
        i_layer: int,
    ) -> dict[str, Tensor]:
        """
        Compute the ReID losses (OIM and Triplet, the former if set).
        For a single decoder layer

        Args:
            all_scales_batch_reid_features (Tensor): ReID features by scale
                and sample. (n_scales, bs, n_queries, n_dim_reid)
            batch_assigned_person_ids (Tensor): Result of the
                the matching from assigner, 0 is not assigned and other value
                the person ID assigned to it (-1 is for detection kept but
                no ReID annotations) (bs, n_queries)
            data_samples (ReIDDetSampleList): List of annotations, length
                equals to batch_size.
            i_layer (int): index of decoder layer.

        Returns:
            dict[str, Tensor]: The keys are the loss based on the input
            (scales) and value is the loss value.
        """
        num_scales = all_scales_batch_reid_features.shape[0]

        all_scales_reid_loss: dict[str, Tensor] = dict()
        for i_scale in reversed(range(num_scales)):
            all_scales_reid_loss |= self.loss_single_layer_scale(
                all_scales_batch_reid_features[i_scale],
                batch_assigned_person_ids,
                data_samples,
                i_layer,
                i_scale,
            )

        return all_scales_reid_loss

    def loss(
        self,
        detection_decoder_states: Tensor,
        references: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        multi_scale_features_maps: Tuple[Tensor],
        all_layers_batch_assigned_person_ids: Tensor,
        data_samples: ReIDDetSampleList,
    ) -> dict[str, Tensor]:
        """
        Compute the ReID losses (OIM and Triplet, the former if set).

        Args:
            detection_decoder_states (Tensor): The outputs of detector which
                have one in inference and 3 (number of decoder layers) in
                training. (n_decoder_layers, bs, n_queries, n_dim_det)
            references (Tensor): References point from deformable attention.
                (n_decoder_layers, bs, n_queries, 4)
            spatial_shapes (Tensor): The dimension of features maps used in
                detection, before flattening. (n_used_detection, 2)
            valid_ratios (Tensor): Valid ratios of detections, in case some are
                annotatated to be ignored. (bs, ratio_dim, 2).
            multi_scale_features_maps (Tuple[Tensor]): The features maps
                from the backbone/neck. A n_scales-tuple, each element
                is (bs, n_dim_neck, x_dim, y_dim).
            all_layers_batch_assigned_person_ids (Tensor): Result of the
                the matching from assigner, 0 is not assigned and other value
                the person ID assigned to it (-1 is for detection kept but no
                ReID annotations) (n_decoder_layers, bs, n_queries)
            data_samples (ReIDDetSampleList): List of annotations, length
                equals to batch_size.

        Returns:
            dict[str, Tensor]: The keys are the loss based on the input
            (scales and decoder outputs) and value is the loss value.
        """

        # (num_layers, num_scales, batch_size, num_queries, n_features)
        all_layers_all_scales_batch_reid_features = torch.stack(
            self.forward(
                detection_decoder_states,
                references,
                spatial_shapes,
                valid_ratios,
                multi_scale_features_maps,
            )).permute((1, 0, 2, 3, 4))

        all_layers_all_scales_batchreid_loss: dict[str, Tensor] = dict()

        num_layers = all_layers_all_scales_batch_reid_features.shape[0]
        for i_layer in reversed(range(num_layers)):
            all_layers_all_scales_batchreid_loss |= self.loss_single_layer(
                all_layers_all_scales_batch_reid_features[i_layer],
                all_layers_batch_assigned_person_ids[i_layer],
                data_samples,
                i_layer,
            )

        return all_layers_all_scales_batchreid_loss
