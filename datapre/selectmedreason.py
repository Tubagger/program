import json
import os
import argparse

def load_medreason(json_path: str, n: int = None):
    """
    读取 MedReason 数据集 (.jsonl 或 .json)
    """
    data = []
    if json_path.endswith(".jsonl"):
        with open(json_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if n is not None and i >= n:
                    break
                data.append(json.loads(line))
    elif json_path.endswith(".json"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if n is not None:
                data = data[:n]
    else:
        raise ValueError("文件必须是 .json 或 .jsonl 格式")
    
    print(f"✅ 已加载 {len(data)} 条样本。")
    return data

def save_medreason(samples, output_file):
    """
    保存 MedReason 数据到 JSON 文件
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"✅ 保存 {len(samples)} 条样本到 {output_file}")


def main():
    parser = argparse.ArgumentParser(description="读取 MedReason 数据集并保存前 N 条样本")
    parser.add_argument("--json_file", type=str, default="../../Dataset/MedReason/ours_quality_33000.jsonl" ,help="输入 MedReason JSONL/JSON 文件路径")
    parser.add_argument("--json_output", type=str, default="../data/MedReason_subset.json", help="输出 JSON 文件路径")
    parser.add_argument("--n", type=int, default=None, help="读取前 N 条样本，默认全部")
    args = parser.parse_args()
    
    samples = load_medreason(args.json_file, n=args.n)
    save_medreason(samples, args.json_output)


if __name__ == "__main__":
    main()
