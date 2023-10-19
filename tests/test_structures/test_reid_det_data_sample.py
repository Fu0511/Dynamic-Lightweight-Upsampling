from typing import runtime_checkable, Protocol
from unittest import TestCase, main

import numpy as np
import pytest
import torch
from mmengine.structures import InstanceData

from mmdet.structures import ReIDDetDataSample
from mmdet.structures.reid_det_data_sample import ReIDDetInstanceData


@runtime_checkable
class CheckableReIDDetInstanceData(ReIDDetInstanceData, Protocol):
    pass


def _equal(a, b):
    if isinstance(a, (torch.Tensor, np.ndarray)):
        return (a == b).all()
    else:
        return a == b


class TestDetDataSample(TestCase):
    def setUp(self):
        self.meta_info = dict(
            img_size=[256, 256],
            scale_factor=np.array([1.5, 1.5]),
            img_shape=torch.rand(4),
        )

        number_instances = 4
        self.instances_data = dict(
            bboxes=torch.rand(number_instances, 4),
            labels=torch.randint(0, 10, (number_instances,), dtype=torch.long),
            reid_labels=torch.randint(0, 10, (number_instances,), dtype=torch.long),
        )
        self.instances = InstanceData(**self.instances_data)

    def test_init(self):
        det_data_sample = ReIDDetDataSample(metainfo=self.meta_info)
        assert "img_size" in det_data_sample
        assert det_data_sample.img_size == [256, 256]
        assert det_data_sample.get("img_size") == [256, 256]

    def test_reid_det_instance_data(self):
        # test gt_instances
        assert isinstance(self.instances, CheckableReIDDetInstanceData)

    def test_setter_gt(self):
        reid_det_data_sample = ReIDDetDataSample()
        reid_det_data_sample.gt_instances = self.instances
        assert "gt_instances" in reid_det_data_sample

        bboxes, labels, reid_labels = (
            reid_det_data_sample.gt_instances.bboxes,
            reid_det_data_sample.gt_instances.labels,
            reid_det_data_sample.gt_instances.reid_labels,
        )
        assert _equal(bboxes, self.instances_data["bboxes"])
        assert _equal(labels, self.instances_data["labels"])
        assert _equal(
            reid_labels,
            self.instances_data["reid_labels"],
        )

    def test_setter_pred(self):
        reid_det_data_sample = ReIDDetDataSample()
        reid_det_data_sample.pred_instances = self.instances
        assert "pred_instances" in reid_det_data_sample

        bboxes, labels, reid_labels = (
            reid_det_data_sample.pred_instances.bboxes,
            reid_det_data_sample.pred_instances.labels,
            reid_det_data_sample.pred_instances.reid_labels,
        )
        assert _equal(bboxes, self.instances_data["bboxes"])
        assert _equal(labels, self.instances_data["labels"])
        assert _equal(
            reid_labels,
            self.instances_data["reid_labels"],
        )

    def test_setter_ignored(self):
        reid_det_data_sample = ReIDDetDataSample()
        reid_det_data_sample.ignored_instances = self.instances
        assert "ignored_instances" in reid_det_data_sample

        bboxes, labels, reid_labels = (
            reid_det_data_sample.ignored_instances.bboxes,
            reid_det_data_sample.ignored_instances.labels,
            reid_det_data_sample.ignored_instances.reid_labels,
        )
        assert _equal(bboxes, self.instances_data["bboxes"])
        assert _equal(labels, self.instances_data["labels"])
        assert _equal(
            reid_labels,
            self.instances_data["reid_labels"],
        )

    def test_type_error(self):
        reid_det_data_sample = ReIDDetDataSample()
        with pytest.raises(AssertionError):
            reid_det_data_sample.gt_instances = torch.rand(2, 4)
        with pytest.raises(AssertionError):
            reid_det_data_sample.pred_instances = torch.rand(2, 4)
        with pytest.raises(AssertionError):
            reid_det_data_sample.ignored_instances = torch.rand(2, 4)

    def test_deleter(self):
        reid_det_data_sample = ReIDDetDataSample()

        reid_det_data_sample.gt_instances = self.instances
        assert "gt_instances" in reid_det_data_sample
        del reid_det_data_sample.gt_instances
        assert "gt_instances" not in reid_det_data_sample

        reid_det_data_sample.pred_instances = self.instances
        assert "pred_instances" in reid_det_data_sample
        del reid_det_data_sample.pred_instances
        assert "pred_instances" not in reid_det_data_sample

        reid_det_data_sample.ignored_instances = self.instances
        assert "ignored_instances" in reid_det_data_sample
        del reid_det_data_sample.ignored_instances
        assert "ignored_instances" not in reid_det_data_sample


if __name__ == "__main__":
    main()
