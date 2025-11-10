import json

def load_medreason(json_path: str, n: int = None):
    """
    读取 MedReason 数据集 (.jsonl 或 .json)
    参数:
        json_path (str): 文件路径
        n (int, 可选): 只读取前 n 条
    """
    data = []
    if json_path.endswith(".jsonl"):
        # 按行读取
        with open(json_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if n is not None and i >= n:
                    break
                data.append(json.loads(line))
    elif json_path.endswith(".json"):
        # 整个文件读取
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if n is not None:
                data = data[:n]
    else:
        raise ValueError("文件必须是 .json 或 .jsonl 格式")
    
    print(f"✅ 已加载 {len(data)} 条样本。")
    return data

# 测试
if __name__ == "__main__":
    dataset_path = "../../Dataset/MedReason/ours_quality_33000.jsonl"
    
    all_data = load_medreason(dataset_path)        # 默认读取全部
    small_data = load_medreason(dataset_path, n=5) # 只读取前 5 条

    import pprint
    pprint.pprint(small_data)
