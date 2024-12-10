import json

def count_keys_values(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        # filtered_data = {k: v for k, v in data.items() if isinstance(v, list) and len(v) > 1}
        key_count = len(data.keys())
        value_count = len(data.values())
        return key_count, value_count, data

if __name__ == "__main__":
    json_file_path = '/home/taehoonlee/AAA740-2024/taehoonlee/LLM-groundedDiffusion/cache/experiment/cache_spatial_v0.2_gpt-4.json'
    
    keys, values, filtered_data = count_keys_values(json_file_path)
    print(f"Number of keys: {keys}")
    print(f"Number of values: {values}")
    print("Filtered data:")
    for k, v in filtered_data.items():
        print(f"{k}: {v}")