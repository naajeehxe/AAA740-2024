import sys
import os
import json
import argparse
import re

words_by_size = [
    "Cup",
    "Bottle",
    "Bowl",
    "Book",
    "Laptop",
    "Backpack",
    "Cat",
    "Dog",
    "Chair",
    "Car"
]

words_by_size = [word.lower() for word in words_by_size]


def bbox_size(bbox):
    return (bbox[2]) * (bbox[3])

def open_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def get_words_by_size(data):
    objects_by_size = {}
    for obj in data["objects"]:
        size = bbox_size(obj["bbox"])
        objects_by_size[obj["id"]] = size
    return objects_by_size

def get_word_bbox_size(data):
    word_box_pair = {}
    matches = re.findall(r"\((.*?)\)", str(data))
    for match in matches:
        obj_name, bbox_str = match.split(", [")
        bbox = list(map(int, bbox_str[:-1].split(", ")))
        # object = obj_name.strip().strip("'")
        word = obj_name.split(" ")[-1].strip("'")
        word_box_pair[word] = bbox_size(bbox)
    return word_box_pair



def check_subset_order(main_list, subset_list):
    index_list = []
    for i in subset_list:
        for j in main_list:
            if i == j:
                # print(i, j)
                # print(main_list.index(i))
                index_list.append(main_list.index(j))
        
    # print(f"index_list:{index_list}")
    # print(sorted(index_list, reverse=False))
    # print(sorted(index_list))
    if index_list == sorted(index_list, reverse=False):
        # print(index_list, sorted(index_list, reverse=True))
        return 1
    else:
        return 0
        
if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--json_path", type=str, required=True)
    args = parser.parse_args()

    # get the keys of the json file.
    data = open_json(args.json_path)
    keys = data.keys()
    
    evaluation_list = []
    # key 만큼의 샘플들이 있고, 이에 대해서 측정해야함.
    for key in keys:
        # print(data[key][0])
        
        #1. {word : size} dictionary 생성.

        word_bbox_pair = get_word_bbox_size(data[key][0])
        # print(word_bbox_pair)

        #2. size 순서대로 정렬.
        sorted_word_bbox_pair = dict(sorted(word_bbox_pair.items(), key=lambda x: x[1], reverse=False))
        # print(sorted_word_bbox_pair)

        # #3. size 순서대로 list 출력.
        sorted_word_list = list(sorted_word_bbox_pair.keys())
        lower_sorted_word_list = [word.lower() for word in sorted_word_list]
        # print(lower_sorted_word_list)


        # #4. 포함되면 list에 append 1, 아니면 0.
        evaluation_list.append(check_subset_order(words_by_size, sorted_word_list))

        # break
    # print(evaluation_list)
    print(f"length: {len(evaluation_list)}")
    print(f"sum: {sum(evaluation_list)}")
    print(f"average: {sum(evaluation_list)/len(evaluation_list)}")    
