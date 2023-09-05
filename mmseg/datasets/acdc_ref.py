from .builder import DATASETS
from .acdc import ACDCDataset


@DATASETS.register_module()
class ACDCRefDataset(ACDCDataset):
    def __init__(self, **kwargs):
        super(ACDCDataset, self).__init__(
            img_suffix='_rgb_ref_anon.png',
            seg_map_suffix='_rgb_ref_anon_labelTrainIds.png',
            **kwargs
        )