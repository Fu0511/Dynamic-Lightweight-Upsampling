from mmengine.dataset import BaseDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class CUHK_SYSU(BaseDataset):
    """
    Dataset for CUHK_SYSU

    The annotations are following the open mmlab annotations so `BaseDataset`
    performs all its operation natively. There are one fix used when loading
    the annotations:

    `self.original_to_adjusted_person_ids` convert non continuous person ids
    from the datasets to a continuous set of IDS. -1 is the ID for detection
    only annotation. This map is used during loading to fix the person ID
    annotations.
    """

    def _create_person_id_index(self) -> None:
        original_person_ids = [
            annotation['person_id'] for data in self.load_data_list()
            for annotation in data['detection_annotations']
            # We only adjust assigned person_ids
            if annotation['person_id'] != -1
        ]

        # NOTE: sorted makes the order consistant between runs.
        unique_sorted_original_person_ids = sorted(set(original_person_ids))
        self.original_to_adjusted_person_ids: dict[int, int] = {
            # +1 because person IDs start at 1
            original_person_id: adjusted_person_id + 1
            for adjusted_person_id, original_person_id in enumerate(
                unique_sorted_original_person_ids)
        }
        # NOTE: Unassigned person ID value stays the same.
        # A model can rely on this -1 value to differentiate detections
        # with or without ReID annotations.
        self.original_to_adjusted_person_ids[-1] = -1

    # kwargs are arguments of the base class.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_person_id_index()

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)

        # Tweak annotations to fit detector models
        for annotation in data_info['detection_annotations']:
            # Adjust the person IDs to be continuous
            annotation['person_id'] = self.original_to_adjusted_person_ids[
                annotation['person_id']]

        return data_info
