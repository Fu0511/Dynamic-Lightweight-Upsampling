from typing import Protocol, runtime_checkable
from torch import LongTensor
from mmengine.structures import BaseDataElement, InstanceData

from mmdet.structures.bbox import HorizontalBoxes

# TODO Remove yapf linting


class ReIDDetInstanceData(Protocol):
    # Detection labels, the name is confusion but we keep to be compatible
    # with existing Detector in mmdet.
    labels: LongTensor
    reid_labels: LongTensor
    bboxes: HorizontalBoxes


class ReIDDetDataSample(BaseDataElement):
    """
    A data structure that represents a sample for ReID Detection task.

    A sample is a frame with multiple instances. There are annotations
    instances (gt_instances), model instances (pred_instances) and
    instances that got filtered (ignore_instances).
    Each InstanceData should respect ReIDDetInstanceData interface.
    This interface is implic

    Data field:
        gt_instances (InstanceData): Ground truth bbox annotations of
            instances.
        pred_instances (InstanceData): bbox predictions from the model.
        ignored_instances (InstanceData): Instances to be ignored
    """

    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, "_gt_instances", dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, "_pred_instances", dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    @property
    def ignored_instances(self) -> InstanceData:
        return self._ignored_instances

    @ignored_instances.setter
    def ignored_instances(self, value: InstanceData):
        self.set_field(value, "_ignored_instances", dtype=InstanceData)

    @ignored_instances.deleter
    def ignored_instances(self):
        del self._ignored_instances
