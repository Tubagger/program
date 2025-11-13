import json
import argparse

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Process Deeplesion samples.")
    # 设置 'N' 参数，默认打印所有样本，用户可以通过命令行传入N值
    parser.add_argument('--n', type=int, default=None, help="Number of samples to print (default: print all)")
    return parser.parse_args()

# 打开并加载 JSON 文件
def load_samples(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['samples']

# 主程序
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载样本数据
    samples = load_samples('../../deeplesion-balanced-2k/samples.json')
    
    # 如果指定了 N，限制只打印前 N 个样本
    n = args.n if args.n is not None else len(samples)  # 如果没有传递 N 参数，打印所有样本
    
    # 打印前 N 个样本的路径和标注信息
    for i, sample in enumerate(samples[:n]):
        image_path = sample['filepath']
        ground_truth = sample['ground_truth']
        
        # 获取所有的病灶检测信息（bounding boxes）
        detections = ground_truth['detections']
        for detection in detections:
            label = detection['label']  # 病灶类型（例如 "lesion"）
            bounding_box = detection['bounding_box']  # 归一化的边界框
            print(f"Sample {i+1}: Image Path: {image_path}, Label: {label}, Bounding Box: {bounding_box}")

if __name__ == "__main__":
    main()