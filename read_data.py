# import pickle

# with open('/home/lhy/audio/ase/LR-ASD/demo/102/pywork/tracks.pckl', 'rb') as f:
#     data = pickle.load(f)

# print(data)

import pickle
import json
import numpy as np

def numpy_to_list(obj):
    # 递归将 numpy 类型转为 list
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(i) for i in obj]
    else:
        return obj

# 修改为你的pckl文件路径
pckl_path = '/home/lhy/audio/ase/LR-ASD/demo/102/pywork/tracks.pckl'
json_path = pckl_path.replace('.pckl', '.json')

with open(pckl_path, 'rb') as f:
    data = pickle.load(f)

# 转换为可json序列化的格式
data_json = numpy_to_list(data)

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data_json, f, ensure_ascii=False, indent=2)

print(f'已保存为 {json_path}')