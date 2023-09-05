""" 
-*- coding: utf-8 -*-
    @Time    : 2023/2/5  20:00
    @Author  : AresDrw
    @File    : learn_dataset.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-

   导入所需要的包：
        注意，一旦被注册进入了openmmlab的registry中
        就不能再通过原有的构造函数进行创建实例了
        ——只能使用带有的build函数进行构建
"""
from mmseg.datasets.builder import build_dataset
from mmseg.datasets.builder import build_dataloader
from mmcv.utils.config import Config

if __name__ == "__main__":

    # 1.普通的数据集：Cityscapes
    # cfg_file = '/hy-tmp/01-DAFormer/configs/_base_/datasets/cityscapes_half_512x512.py'
    # 2. UDA数据集：cityscapes->ACDC
    cfg_file = '/hy-tmp/01-DAFormer/configs/_base_/datasets/uda_cityscapes_to_acdc_512x512.py'

    # 标准化的数据读入接口
    """
        Dataset是"数据库"
        DataLoader是将dataset里的数据一个一个喂给模型的工具
        大家要熟悉基于PyTorch的这种标准化做法
            i.e., 使用DataLoader取出数据
    """
    data_cfg = Config.fromfile(cfg_file)
    cs_dataset = build_dataset(data_cfg.data.train)  # must have 'type'
    cs_dataloader = build_dataloader(dataset=cs_dataset,
                                     samples_per_gpu=2,
                                     workers_per_gpu=1)

    for i, data in enumerate(cs_dataloader):
        if i == 0:
            if 'uda' not in cfg_file:
                """
                    1、普通的数据集最重要的就是3个字段：
                        |--img：图像数据；Tensor: [B, C, H, W] e.g., [2, 3, 512, 512]
                        |--img_metas: 图像信息；str
                        |--gt_semantic_seg: 标签； [B, 1, H, W] e.g., [2, 1, 512, 512]
                """
                print('data.img_metas:', data['img_metas'].data[0][0]['filename'])
                print('gt', data['gt_semantic_seg'].data[0])
                print('gt.size', data['gt_semantic_seg'].data[0].size())
            else:
                """
                    2、UDA数据集(源域、目标域）的5个字段
                        |--img：源域图像
                        |--img_metas: 源域图像信息
                        |--gt_segmantic_seg: 源域标签
                        |--target_img: 目标域图像
                        |--target_img_metas: 目标域图像信息
                    思考：
                        (1) 为什么目标域加载了ann_dir数据集但没有gt_segmantic_seg?
                        (2) 目标域需要标签吗？
                        (3) 我们要如何来配置这样的效果呢？
                        (4) 能不能加入更多的域信息？
                """
            print('data.img_metas:', data['img_metas'].data[0][0]['filename'])
            print('gt', data['gt_semantic_seg'].data[0])
            print('gt.size', data['gt_semantic_seg'].data[0].size())
            print('data.target_img_metas:', data['img_metas'].data[0][0]['filename'])
            break
