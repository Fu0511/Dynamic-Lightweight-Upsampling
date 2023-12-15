"""
Source code from PSTR
https://github.com/JialeCao001/PSTR/blob/main/mmdet/models/dense_heads/reid_matching_layer.py
"""
import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Function


class LabeledMatchingQ(Function):

    @staticmethod
    def forward(ctx,
                features: Tensor,
                pid_labels: Tensor,
                lookup_table: Tensor,
                momentum: float = 0.5):
        # The lookup_table can't be saved with ctx.save_for_backward(), as we
        # would modify the variable which has the same memory address in
        # backward()

        ctx.save_for_backward(features, pid_labels)
        ctx.lookup_table = lookup_table
        ctx.momentum = momentum

        scores = features.mm(lookup_table.t())

        positive_indexes = pid_labels > 0
        positive_person_ids = pid_labels[positive_indexes]

        # -1 because person_ids are 1-indexed but lookup_table is 0-indexed.
        positive_features = lookup_table.clone().detach()[positive_person_ids -
                                                          1]

        return scores, positive_features, positive_person_ids

    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)

        # Update lookup table, but not by standard backpropagation with
        # gradients.
        # NOTE: Person IDs starts at 1 for other region of code.
        # Here we should reduce by one so it fits in the table.
        for indx, label in enumerate(pid_labels - 1):
            if label >= 0:
                lookup_table[label] = (
                    momentum * lookup_table[label] +
                    (1 - momentum) * features[indx])

        return grad_feats, None, None, None


class LabeledMatchingLayerQ(nn.Module):
    """
    Labeled matching of OIM loss function.
    """

    def __init__(self, num_persons: int = 5532, feat_len: int = 256):
        """
        Args:
            num_persons (int): Number of labeled persons.
            feat_len (int): Length of the feature extracted by the network.
        """
        super(LabeledMatchingLayerQ, self).__init__()
        self.register_buffer("lookup_table",
                             torch.zeros(num_persons, feat_len))

    def forward(self, features: Tensor, pid_labels: Tensor):
        """
        Args:
            features (Tensor[N, feat_len]): Features of the proposals.
            pid_labels (Tensor[N]): Ground-truth person IDs of the proposals.

        Returns:
            scores (Tensor[N, num_persons]): Labeled matching scores, namely
                the similarities between proposals and labeled persons.
        """

        scores, pos_feats, pos_pids = LabeledMatchingQ.apply(
            features, pid_labels, self.lookup_table)
        return scores, pos_feats, pos_pids


class UnlabeledMatching(Function):

    @staticmethod
    def forward(ctx, features, pid_labels, queue, tail):
        # The queue/tail can't be saved with ctx.save_for_backward(), as we
        # would modify the variable which has the same memory address in
        # backward().
        ctx.save_for_backward(features, pid_labels)
        ctx.queue = queue
        ctx.tail = tail

        scores = features.mm(queue.t())

        return scores

    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        queue = ctx.queue
        tail = ctx.tail

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(queue.data)

        # # Update circular queue, but not by standard backpropagation with
        # gradients
        for indx, label in enumerate(pid_labels):
            if label == -1:
                queue[tail, :] = features[indx, :]
                tail += 1
                if tail >= queue.size(0):
                    tail -= queue.size(0)
        return grad_feats, None, None, None


class UnlabeledMatchingLayer(nn.Module):
    """
    Unlabeled matching of OIM loss function.
    """

    def __init__(self, queue_size=5000, feat_len=256):
        """
        Args:
            queue_size (int): Size of the queue saving the features of
                unlabeled persons.
            feat_len (int): Length of the feature extracted by the network.
        """
        super(UnlabeledMatchingLayer, self).__init__()
        self.register_buffer("queue", torch.zeros(queue_size, feat_len))
        self.register_buffer("tail", torch.tensor(0))

    def forward(self, features, pid_labels):
        """
        Args:
            features (Tensor[N, feat_len]): Features of the proposals.
            pid_labels (Tensor[N]): Ground-truth person IDs of the proposals.

        Returns:
            scores (Tensor[N, queue_size]): Unlabeled matching scores, namely
                the similarities between proposals and unlabeled persons.
        """
        scores = UnlabeledMatching.apply(features, pid_labels, self.queue,
                                         self.tail)
        return scores
