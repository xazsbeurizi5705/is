# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.models.uda.dacs import DACS
from mmseg.models.uda.multi_teacher_IMDTGT_mix_trainchange import MultiTeacherIMDTGT
#修改文件位置更换为mmseg/models/uda/MultiTeacherIMDTGT or mmseg/models/uda/multi_teacher_IMDTGT_mix(不同的mix方式)
#修改文件位置更换为mmseg/models/uda/multi_teacher_IMDTGT_mix_trainchange(不同的训练策略，交叉还是不交叉)
__all__ = ['DACS',
           'Multi_Teacher_IMDTGT']
