import sys
import os
import json
import argparse
import re

objects_by_size = [
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


def bbox_size(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def open_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def get_objects_by_size(data):
    objects_by_size = {}
    for obj in data["objects"]:
        size = bbox_size(obj["bbox"])
        objects_by_size[obj["id"]] = size
    return objects_by_size

if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--json_path", type=str, required=True)
    args = parser.parse_args()

    # get the keys of the json file.
    data = open_json(args.json_path)
    keys = data.keys()

    for key in keys:
        # print(data[key][0])
        matches = re.findall(r"\((.*?)\)", str(data[key][0]))
        for match in matches:
            obj_name, bbox_str = match.split(", [")
            bbox = list(map(int, bbox_str[:-1].split(", ")))
            object = obj_name.strip().strip("'")
            print(f"Object: {object}  , BBox: {bbox}")        
        break    
    

