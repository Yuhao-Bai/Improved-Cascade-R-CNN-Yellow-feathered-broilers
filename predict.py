import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, write_bbox_result

import os
from tqdm import tqdm

cfg = mmcv.Config.fromfile('configs/faster_rcnn_r101_fpn_1x.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model_weight_path = '/home/njaudata/anaconda3/envs/mmdetection/mmdetection/weight/faster_rcnn_r106/latest.pth'
data_root = './defect/'
data_dir = 'images/'
output_dir = 'defect_txt/val'
print(output_dir)
print(model_weight_path)
# model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
# _ = load_checkpoint(model, model_weight_path)
model = init_detector(cfg, model_weight_path)
file_list = os.listdir(os.path.join(data_root,data_dir))
file_list.sort()
for image_file in tqdm(file_list):
    image_name = os.path.join(data_root, data_dir, image_file)
    img = mmcv.imread(image_name)
    result = inference_detector(model, img)
    write_bbox_result(image_file, result, dataset='coco',
    score_thr=0., out_file=output_dir)
