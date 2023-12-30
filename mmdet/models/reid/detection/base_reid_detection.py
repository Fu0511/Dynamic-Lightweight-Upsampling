from abc import ABCMeta, abstractmethod
from typing import TypeAlias

from mmengine.model import BaseModel
from torch import Tensor

from mmdet.models.detectors import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures.reid_det_data_sample import (OptReIDDetSampleList,
                                                   ReIDDetInstanceData,
                                                   ReIDDetSampleList)
from mmdet.utils import ConfigType, OptConfigType
from mmdet.utils.typing_utils import InstanceList

RawResult: TypeAlias = tuple[Tensor, ...] | Tensor
LossResult: TypeAlias = dict[str, Tensor]
PredictResult: TypeAlias = InstanceList
ForwardResult: TypeAlias = RawResult | LossResult | PredictResult


@MODELS.register_module()
class BaseReIDDetection(BaseModel, metaclass=ABCMeta):
    """
    Base class for ReID Detection model. This base class aims to implement
    some methods and indicate the overall usage of ReID Detection model.

    The method extract_feat and forward are implemented. The extract_feat uses
    the backbone and the neck of the detector. The forward method is the same
    as the BaseDetector method.

    Args:
        detector (ConfigType): The config file for the detector.
        reid (HeadModel): The config file for the ReID model.
        train_cfg (OptConfigType): The training config.
        test_cfg (OptConfigType): The testing config.
    """

    def __init__(
        self,
        detector: ConfigType,
        reid: ConfigType,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        detector.update(train_cfg=train_cfg)
        detector.update(test_cfg=test_cfg)

        self.detector: BaseDetector = MODELS.build(detector)
        self.reid = MODELS.build(reid)

    def extract_feat(self, batch_inputs: Tensor) -> tuple[Tensor, ...]:
        """Extract features from the detector

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = self.detector.backbone(batch_inputs)
        if self.detector.neck:
            x = self.detector.neck(x)
        return x

    def forward(
        self,
        inputs: Tensor,
        data_samples: OptReIDDetSampleList = None,
        mode: str = "tensor",
    ) -> ForwardResult:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`ReIDDataSampleList`], optional): A batch
                of data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == "loss" and data_samples:
            return self.loss(inputs, data_samples)
        elif mode == "predict" and data_samples:
            return self.predict(inputs, data_samples)
        elif mode == "tensor":
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               "Only supports loss, predict and tensor mode")

    @abstractmethod
    def loss(
        self,
        inputs: Tensor,
        data_samples: ReIDDetSampleList,
    ) -> LossResult:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    @abstractmethod
    def predict(self, inputs: Tensor,
                data_samples: ReIDDetSampleList) -> list[ReIDDetInstanceData]:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        pass

    @abstractmethod
    def _forward(
        self,
        inputs: Tensor,
        data_samples: OptReIDDetSampleList = None,
    ) -> RawResult:
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass
