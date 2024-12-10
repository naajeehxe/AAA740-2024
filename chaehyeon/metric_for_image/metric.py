import os, json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import combinations
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--src_pth', type=str, default="results/sample")
parser.add_argument('--words_by_size', type=str, default="['Cup', 'Bottle', 'Bowl', 'Book', 'Laptop', 'Backpack', 'Cat', 'Dog', 'Chair', 'Car']")
parser.add_argument('--_INPUT_RESOLUTION', type=int, default=512)
args = parser.parse_args()

from annotator.midas import MidasDetector # midas depth
apply_dep = MidasDetector()
os.makedirs(os.path.join(args.src_pth, "img_gen_dep"), exist_ok=True)
os.makedirs(os.path.join(args.src_pth, "img_gen_dep_info"), exist_ok=True)

words_by_size = args.words_by_size.replace("[", "").replace("]", "").replace("'", "").replace(" ", "").split(",")
words_by_size = [value.lower() for value in words_by_size] ## 모두 소문자로 처리

info_json = json.load(open(os.path.join(args.src_pth, 'info.json')))
final_metric_results = {'details' : {}}
for key_imgpth in tqdm(info_json):
    img = np.array(Image.open(key_imgpth))
    img_info = info_json[key_imgpth]
    img_info['category_list'] = [value.lower() for value in img_info['category_list']] ## 모두 소문자로 처리

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
        text_y = metric_results[cate_info]['original_bbox'][1] - 5 # 텍스트의 y 좌표 (bbox 위에 약간 띄워서 표시)
        ax.text(text_x, text_y,
                f"[{cate_info}] {metric_results[cate_info]['original_depth']:.2f}",
                fontsize=8, color='yellow', 
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
        )

    fig.savefig(key_imgpth.replace("img_gen", "img_gen_dep_info"))
    plt.close()

    ## TODO : 가능한 모든 한 쌍들의 조합에 대해, 통념에 부합하는지 확인
    if len(metric_results.keys()) == 1: ## case (a) : 이미지에 존재하는 object가 1개인 경우 == 비교할 것이 없음
        pass
    else: ## case (b & c) : 이미지에 존재하는 object가 여러개인 경우 == 모든 nC2 조합에 대해 비교해봐야 함
        pairs = list(combinations(list(metric_results.keys()), 2))
        for pair1, pair2 in pairs:
            if pair1.split('#')[0] == pair2.split('#')[0]: ## case (b) : 서로 같은 object 간의 조합인 경우 == 비교할 것이 없음
                pass
            else: ## case (c) : 서로 다른 object 간의 조합인 경우 == 비교 필요
                former_bbox_is_wider = metric_results[pair1]['revised_bbox_area'] > metric_results[pair2]['revised_bbox_area']
                former_cate_is_wider = words_by_size.index(pair1.split('#')[0]) >  words_by_size.index(pair2.split('#')[0])
                if key_imgpth.split("/")[-1] not in final_metric_results['details'].keys(): final_metric_results['details'][key_imgpth.split("/")[-1]] = {}
                final_metric_results['details'][key_imgpth.split("/")[-1]][f"{pair1}_{pair2}"] = 'O' if former_bbox_is_wider == former_cate_is_wider else 'X'

        with open(key_imgpth.replace("img_gen", "img_gen_dep_info").replace("png", "json"), 'w') as f1:
            json.dump(metric_results, f1, indent=4)

accuracy_for_total = []
for temp1 in final_metric_results['details'].keys():
    result_for_img = []
    for temp2 in final_metric_results['details'][temp1].keys():
        result_for_img.append(final_metric_results['details'][temp1][temp2])
    accuracy_for_img = result_for_img.count('O')/len(result_for_img)
    final_metric_results['details'][temp1]['accuracy_for_img'] = accuracy_for_img
    accuracy_for_total.append(accuracy_for_img)
final_metric_results['accuracy_for_total_imgs'] = np.sum(accuracy_for_total)/len(accuracy_for_total)

with open(os.path.join(args.src_pth, 'final_metric_results.json'), 'w') as f2:
    json.dump(final_metric_results, f2, indent=4)