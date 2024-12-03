import os, json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--src_pth', type=str, default="results/sample")
parser.add_argument('--_INPUT_RESOLUTION', type=int, default=512)
args = parser.parse_args()

from annotator.midas import MidasDetector # midas depth
apply_dep = MidasDetector()
os.makedirs(os.path.join(args.src_pth, "img_gen_dep"), exist_ok=True)
os.makedirs(os.path.join(args.src_pth, "img_gen_dep_info"), exist_ok=True)

info_json = json.load(open(os.path.join(args.src_pth, 'info.json')))
for key_imgpth in tqdm(info_json):
    img = np.array(Image.open(key_imgpth))
    img_info = info_json[key_imgpth]
    with torch.no_grad():
        img_dep = apply_dep(img)
        Image.fromarray(img_dep).save(key_imgpth.replace("img_gen", "img_gen_dep"))
    
    if len(img_info['category_list']) != len(list(set(img_info['category_list']))): ## NOTE : category_list에 중복되는 요소가 있는 경우
        # 중복인 category를 찾고, 중복인 category들에 #1, #2 등으로 붙여줌
        categories = sorted(list(set(img_info['category_list']))) ## NOTE : 존재하는 모든 cateogory를 알아냄
        for _cate_ in categories:
            _cnt_ = img_info['category_list'].count(_cate_) ## NOTE : 각 category가 몇 개씩 있는지 확인
            if _cnt_ == 1: ## NOTE : 1개면 그냥 지나감
                pass
            else: ## NOTE : 1개가 아니라면 (중복이 있다면)
                _idx_ = np.where(np.array(img_info['category_list']) == _cate_)[0] ## NOTE : 중복되는 요소의 index를 찾음
                flag = 1
                for _i_ in _idx_:
                    img_info['category_list'][_i_] = img_info['category_list'][_i_] + f"#{flag}"
                    flag += 1
                assert flag-1 == _cnt_  

    metric_results = {}
    fig, ax = plt.subplots()
    ax.imshow(Image.fromarray(img_dep))
    for bbox_info, cate_info in zip(img_info['bbox_list'], img_info['category_list']):
        dep_per_obj = img_dep[bbox_info[1]:bbox_info[1]+bbox_info[3], bbox_info[0]:bbox_info[0]+bbox_info[2]]
        obj_dep_ori = float(np.mean(dep_per_obj))
        bbox_rev = [bbox_info[3]/obj_dep_ori, bbox_info[2]/obj_dep_ori]
        metric_results[cate_info] = {'original_depth' : obj_dep_ori,
                                     'original_bbox' : bbox_info,
                                     'revised_bbox' : bbox_rev,
                                     'revised_bbox_area' : bbox_rev[0]*bbox_rev[1]}

    area_sum = 0
    for cate_info in metric_results: area_sum += metric_results[cate_info]['revised_bbox_area']
    for cate_info in metric_results:
        metric_results[cate_info]['revised_bbox_area'] /= area_sum
    for cate_info in metric_results:
        ## bounding box
        rect = patches.Rectangle(
            (metric_results[cate_info]['original_bbox'][0], metric_results[cate_info]['original_bbox'][1]), # (x, y) (좌상단)
            metric_results[cate_info]['original_bbox'][2], metric_results[cate_info]['original_bbox'][3], # (width, height)
            linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ## text about bounding box
        text_x = metric_results[cate_info]['original_bbox'][0] # 텍스트의 x 좌표
        text_y = metric_results[cate_info]['original_bbox'][1] - 5  # 텍스트의 y 좌표 (bbox 위에 약간 띄워서 표시)
        ax.text(text_x, text_y,
                f"[{cate_info}] {metric_results[cate_info]['original_depth']:.2f}",
                fontsize=8, color='yellow', 
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
        )

    fig.savefig(key_imgpth.replace("img_gen", "img_gen_dep_info"))
    plt.close()
    with open(key_imgpth.replace("img_gen", "img_gen_dep_info").replace("png", "json"), 'w') as f:
        json.dump(metric_results, f, indent=4)

import pdb; pdb.set_trace()