# -*- coding: utf-8 -*-
# @Author  : PuQing
# @Time    : 2021-09-05 12:04
# @File : eva.py
import paddle
from paddleseg.core import evaluate
from models.model_stages import CPNet
from paddleseg.datasets import ADE20K
import paddleseg.transforms as T

backbonepath = None
model = CPNet(proir_size=60, am_kernel_size=11, groups=1, prior_channels=256, pretrained=backbonepath)
model.set_state_dict(paddle.load('/home/aistudio/work/openContext/output/best_model/model.pdparams'))

transform_val = [
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
]

val_dataset = ADE20K(
    dataset_root='/home/aistudio/data/data54455/ADEChallengeData2016',
    transforms=transform_val,
    mode='val'
)

mean_iou, acc, _, _, _ = evaluate(
    model,
    val_dataset,
    aug_eval=True,
    scales=[1.0, 1.5, 1.75],
    num_workers=0)
