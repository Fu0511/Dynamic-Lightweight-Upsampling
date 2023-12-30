import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models import DeformableDETR
from mmdet.models.reid.detection.base_reid_detection import BaseReIDDetection
from mmdet.models.reid_heads import PSTRHeadReID
from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.structures.reid_det_data_sample import ReIDDetInstanceData


@MODELS.register_module()
class PSTR(BaseReIDDetection):
    """
    Implémentation of CVPR22 PSTR paper.
    https://arxiv.org/abs/2204.03340

    This is not the official implementation, the official implementation has
    been ported from mmdet2 to mmdet3.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.detector: DeformableDETR
        self.reid: PSTRHeadReID

    def forward_detector_no_head(
        self,
        multi_scale_features_maps: tuple[Tensor, ...],
        data_samples: OptSampleList = None,
    ) -> tuple[dict, dict]:
        """
        Similar of forward_transformer of DetectionTransformer base class.

        We could not use the whole forward from the detector since we need
        some additional outputs from the detector for the ReID head.

        Args:
            multi_scale_features_maps (tuple[Tensor, ...]): features maps from
                the neck with multiple scale, one scale by features map.
            data_samples (OptSampleList, optional): detections compliant data
                sample. Defaults to None.

        Returns:
            tuple[dict, dict]: 2-tuple of the dicitonnaries for the input of
                the detection and reid heads.
        """
        # PSTR takes only latest features map for detection
        # NOTE Latest features map is first because the neck operation
        encoder_decoder_inputs_dicts = self.detector.pre_transformer(
            (multi_scale_features_maps[0], ), data_samples)
        encoder_inputs: dict = encoder_decoder_inputs_dicts[0]
        decoder_inputs: dict = encoder_decoder_inputs_dicts[1]

        encoder_outputs = self.detector.forward_encoder(**encoder_inputs)
        tmp_dec_in, detection_head_inputs = self.detector.pre_decoder(
            **encoder_outputs)
        decoder_inputs.update(tmp_dec_in)

        decoder_outputs = self.detector.forward_decoder(**decoder_inputs)
        detection_head_inputs.update(decoder_outputs)

        # Format ReID inputs
        reid_decoder_inputs = dict(
            multi_scale_features_maps=multi_scale_features_maps,
            detection_decoder_states=decoder_outputs["hidden_states"],
            valid_ratios=encoder_inputs["valid_ratios"],
            spatial_shapes=encoder_inputs["spatial_shapes"],
        )

        return (
            detection_head_inputs,
            reid_decoder_inputs,
        )

    def _forward(
        self,
        inputs: Tensor,
        data_samples: SampleList,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        multi_scale_features_maps = self.extract_feat(inputs)

        (
            detection_head_inputs,
            reid_decoder_inputs,
        ) = self.forward_detector_no_head(multi_scale_features_maps,
                                          data_samples)

        all_layers_outputs_classes: Tensor
        all_layers_outputs_coords: Tensor
        (
            all_layers_outputs_classes,
            all_layers_outputs_coords,
        ) = self.detector.bbox_head.forward(**detection_head_inputs)
        reid_decoder_inputs["references"] = all_layers_outputs_coords

        all_layers_reid_features = self.reid.forward(**reid_decoder_inputs)

        return_value = (
            all_layers_outputs_classes,
            all_layers_outputs_coords,
            all_layers_reid_features,
        )
        return return_value

    def predict(self, inputs: Tensor,
                data_samples: SampleList) -> list[ReIDDetInstanceData]:
        return self._forward(inputs, data_samples)

    def _get_targets_single_layer_sample(
        self,
        cls_pred: Tensor,
        bbox_pred: Tensor,
        gt_instances: ReIDDetInstanceData,
        img_metas: dict,
    ) -> Tensor:
        """
        Perform _get_targets_single_layer for a single batch.

        Args:
            outputs_classes (Tensor): Output of the classification
                head of the detector.
                (n_queries, n_det_class = 1)
            outputs_coords (Tensor): Output of the regression
                head of the detector. (n_queries, bbox_dim = 4)
            batch_gt_instances (ReIDDetInstanceData): Ground truth
                instances (bboxes) on the current frames.
            batch_img_metas (dict): Metainformation of the batch frames.

        Returns:
            Tensor: Assigned person_ids for query, batch size.
                Value is 0 if not assigned, else has the person id as value.
                -1 for detections with no person ids. (n_queries)
        """

        assigner: BaseAssigner = self.detector.bbox_head.assigner

        img_h, img_w = img_metas["img_shape"]

        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pred_instances = InstanceData(
            scores=cls_pred,
            # convert bbox_pred from xywh, normalized to xyxy, unnormalized
            bboxes=bbox_cxcywh_to_xyxy(bbox_pred) * factor)

        # assigner and sampler
        assign_result: AssignResult = assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_metas)

        # element is 0 if not assigned, there is no person ID 0
        # -1 or another integer for person_id
        assigned_person_ids = torch.tensor(
            [
                i if i == 0 else
                gt_instances.reid_labels[i - 1]  # gt.inds is 1-indexed
                for i in assign_result.gt_inds
            ],
            device=bbox_pred.device).long()

        # (num_queries)
        return assigned_person_ids

    def _get_targets_single_layer(
        self,
        outputs_classes: Tensor,
        outputs_coords: Tensor,
        batch_gt_instances: list[ReIDDetInstanceData],
        batch_img_metas: list[dict],
    ) -> Tensor:
        """
        Perform _get_targets for a single output of the decoder.

        Args:
            outputs_classes (Tensor): Output of the classification
                head of the detector.
                (bs, n_queries, n_det_class = 1)
            outputs_coords (Tensor): Output of the regression
                head of the detector. (bs, n_queries, bbox_dim = 4)
            batch_gt_instances (list[ReIDDetInstanceData]): Ground truth
                instances (bboxes) on the current frames. [bs]
            batch_img_metas (list[dict]): Metainformation of the batch frames.
                [bs]

        Returns:
            Tensor: Assigned person_ids for query, batch size.
                Value is 0 if not assigned, else has the person id as value.
                -1 for detections with no person ids. (bs, n_queries)
        """
        # NOTE -2 is false positive, -1 is unlabeled detection, rest is labeled
        # For each sample, map the choosen detection with its annotations
        # retrieve the person id from batch_gt_instances
        batch_size = len(batch_gt_instances)
        assert (batch_size == len(batch_img_metas) == outputs_coords.shape[0]
                == outputs_classes.shape[0])

        batch_assigned_person_ids = [
            self._get_targets_single_layer_sample(
                outputs_classes[i_sample],
                outputs_coords[i_sample],
                batch_gt_instances[i_sample],
                batch_img_metas[i_sample],
            ) for i_sample in range(batch_size)
        ]

        # (batch_size, num_queries)
        return torch.stack(batch_assigned_person_ids)

    def _get_targets(
        self,
        all_layers_outputs_classes: Tensor,
        all_layers_outputs_coords: Tensor,
        batch_gt_instances: list[ReIDDetInstanceData],
        batch_img_metas: list[dict],
    ) -> Tensor:
        """
        Similar to the get_targets method from detectors. It has to assign
        the "targets" - detections picked to compute reid features and losses -
        based on an assigner.

        Args:
            all_layers_outputs_classes (Tensor): Output of the classification
                head of the detector.
                (n_layers, bs, n_queries, n_det_class = 1)
            all_layers_outputs_coords (Tensor): Output of the regression
                head of the detector. (n_layers, bs, n_queries, bbox_dim = 4)
            batch_gt_instances (list[ReIDDetInstanceData]): Ground truth
                instances (bboxes) on the current frames. [bs]
            batch_img_metas (list[dict]): Metainformation of the batch frames.
                [bs]

        Returns:
            Tensor: Assigned person_ids for decoder_layer, query, batch size.
                Value is 0 if not assigned, else has the person id as value.
                -1 for detections with no person ids. (n_layers, bs, n_queries)
        """
        num_layers = all_layers_outputs_classes.shape[0]
        assert num_layers == all_layers_outputs_coords.shape[0]

        all_layers_batch_assigned_person_ids = [
            self._get_targets_single_layer(all_layers_outputs_classes[i_layer],
                                           all_layers_outputs_coords[i_layer],
                                           batch_gt_instances, batch_img_metas)
            for i_layer in range(num_layers)
        ]

        # (num_layers, batch_size, num_queries)
        return torch.stack(all_layers_batch_assigned_person_ids)

    def loss(self, inputs: Tensor,
             data_samples: SampleList) -> dict[str, Tensor]:
        # Prepare instances variables
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        # Start forward
        multi_scale_features_maps = self.extract_feat(inputs)

        (
            detection_head_inputs,
            reid_loss_inputs,
        ) = self.forward_detector_no_head(multi_scale_features_maps,
                                          data_samples)

        unsupported_keys = ["enc_outputs_class", "enc_outputs_coord"]
        detection_head_inputs = {
            key: value
            for key, value in detection_head_inputs.items()
            if key not in unsupported_keys
        }

        # Get intermediate output
        all_layers_outputs_classes: Tensor
        all_layers_outputs_coords: Tensor
        (
            all_layers_outputs_classes,
            all_layers_outputs_coords,
        ) = self.detector.bbox_head.forward(**detection_head_inputs)
        reid_loss_inputs["references"] = all_layers_outputs_coords

        # None value are the traces of enc_outputs_class and enc_outputs_coord
        # keys. PSTR does not support these keys.
        detection_loss_inputs = (
            all_layers_outputs_classes,
            all_layers_outputs_coords,
            None,
            None,
            batch_gt_instances,
            batch_img_metas,
        )
        detection_losses = self.detector.bbox_head.loss_by_feat(
            *detection_loss_inputs)

        reid_loss_inputs[
            "all_layers_batch_assigned_person_ids"] = self._get_targets(
                all_layers_outputs_classes,
                all_layers_outputs_coords,
                batch_gt_instances,
                batch_img_metas,
            )
        reid_loss_inputs["data_samples"] = data_samples
        reid_losses = self.reid.loss(**reid_loss_inputs)

        return detection_losses | reid_losses
