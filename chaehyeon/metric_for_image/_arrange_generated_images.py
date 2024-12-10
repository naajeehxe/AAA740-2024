import os, json, re, shutil
from tqdm import tqdm

coco10 = [value.lower() for value in ['Cup', 'Bottle', 'Bowl', 'Book', 'Laptop', 'Backpack', 'Cat', 'Dog', 'Chair', 'Car']]

src_pth = "/home/chaehyeon/code_answer_with_images/metric_for_image/results/img_generations_templatev0.2_lmd_plus_lmd_attribution_gpt-4/run0"
src_json_pth = "/home/chaehyeon/code_answer_with_images/metric_for_image/results/img_generations_templatev0.2_lmd_plus_lmd_attribution_gpt-4/run0/cache_attribution_v0.2_gpt-4.json"
dst_pth = "results/sample_by_taehoon"
os.makedirs(dst_pth, exist_ok=True)
os.makedirs(os.path.join(dst_pth, 'img_gen'), exist_ok=True)

idx_dirs = [value for value in os.listdir(src_pth) if not value.endswith('.json')]
idx_dirs = sorted(idx_dirs, key=lambda x:int(x))

json_ori = json.load(open(src_json_pth))
json_to_save = {}

for idx_dir, key_for_json in zip(tqdm(idx_dirs), list(json_ori.keys())):
    dir_files = sorted(os.listdir(os.path.join(src_pth, idx_dir)))
    
    # assert len(dir_files) == 2, f"### NOTE : 파일이 2개가 아니므로 코드 점검 필요 : {idx_dir, dir_files}"
    if len(dir_files) != 2: 
        print(f"### PASS : dir_files")
        continue

    new_fname = '0' * (4-len(str(int(idx_dir)+1))) + str(int(idx_dir)+1) + '.png'
    src_file_path = os.path.join(src_pth, idx_dir, 'img_0.png') # 복사할 파일
    dst_file_path = os.path.join(dst_pth, 'img_gen', new_fname) # 복사 위치 및 파일 이름 지정
    shutil.copyfile(src_file_path, dst_file_path)

    assert len(json_ori[key_for_json]) == 1, f"### NOTE : 한 caption에 한 bbox prompt가 아니므로 코드 점검 필요 : {dir_files}"
    matches = re.findall(r"\((.*?)\)", json_ori[key_for_json][0].split("\n")[0].replace("[", "").replace("]", "")) # 정규표현식으로 데이터를 추출
    matches = [eval(f"({match})") for match in matches]
    _bbox_list, _cate_list = [], []
    for match in matches:
        _cate = match[0].split(" ")[-1]
        assert _cate in coco10
        _bbox_list.append(list(match[1:]))
        _cate_list.append(_cate)
    
    import pdb; pdb.set_trace()

    if key_for_json == "A realistic photo of a scene with a red dog and a gray laptop":
        import pdb; pdb.set_trace()
    
    json_to_save[os.path.join(dst_pth, 'img_gen', new_fname)] = {'bbox_list' : _bbox_list, 'category_list' : _cate_list}

with open(os.path.join(dst_pth, 'info.json'), 'w') as f:
    json.dump(json_to_save, f, indent=4)