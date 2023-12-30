from typing import Protocol

from mmengine.structures import BaseDataElement, InstanceData
from torch import LongTensor

from mmdet.structures.bbox import HorizontalBoxes


class ReIDDetInstanceData(Protocol):
    """
    This is the implicit type annotations for the ReIDDetDataSample. For now
    we only statically present the InstanceData object from ReID detection
    tasks to have those fields
    NOTE: ReIDDetInstanceData objects are not statically InstanceData. The
    BaseReIDDetection class has its predict method returning this protocol
    instead of a classic list of InstanceData which is not consistent with
    the rest of mmdet. We might update the annotations from this base method
    later and keep this protocol as implicit.
    """
    # Detection labels, the name is confusing but we keep to be compatible
    # with existing Detector in mmdet.
    labels: LongTensor
    #: The person IDs
    reid_labels: LongTensor
    #: The detections
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


ReIDDetSampleList = list[ReIDDetDataSample]
OptReIDDetSampleList = ReIDDetSampleList | None
