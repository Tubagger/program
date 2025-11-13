import os
import pandas as pd
from PIL import Image
import argparse

def load_chexpert_data(csv_file, images_root, n=None):
    """
    加载 CheXpert 数据集的图像路径与对应标签。
    参数:
        csv_file: 标注文件路径（train.csv 或 valid.csv）
        images_root: 数据集根目录
        n: 只加载前 N 条（None 表示加载全部）
    返回:
        records: list(dict)，每个元素包含 {path, image, label_dict}
    """
    df = pd.read_csv(csv_file)
    print(f"📄 CSV 文件共 {len(df)} 条记录。")

    if n is not None:
        df = df.head(n)

    label_columns = df.columns[5:]  # 从第6列开始是标签列
    records = []

    for index, row in df.iterrows():
        image_rel_path = row["Path"]

        # 拼接路径（相对路径补全）
        image_full_path = (
            image_rel_path
            if os.path.isabs(image_rel_path)
            else os.path.normpath(os.path.join(images_root, image_rel_path))
        )

        # 构建标签字典
        label_dict = {label: row[label] for label in label_columns}

        try:
            image = Image.open(image_full_path)
            records.append({
                "path": image_full_path,
                "image": image,
                "labels": label_dict
            })
        except Exception as e:
            print(f"⚠️ 无法加载图像 {image_full_path}: {e}")

    return records
def main():
    parser = argparse.ArgumentParser(description="CheXpert 数据查看工具")
    parser.add_argument("--csv", type=str, default="../../CheXpert-v1.0-small/train.csv", help="CSV 文件路径")
    parser.add_argument("--images", type=str, default="../../", help="图像根目录")
    parser.add_argument("--n", type=int, default=None, help="显示前 N 张图像（默认显示全部）")
    args = parser.parse_args()


    records = load_chexpert_data(args.csv, args.images, args.n)

    print(f"\n✅ 成功加载 {len(records)} 条数据。\n")
    for i, rec in enumerate(records):
        print(f"[{i+1}] {rec['path']}")
        print("Labels:")
        for k, v in rec["labels"].items():
            if not pd.isna(v):  # 只打印有值的标签
                print(f"  - {k}: {v}")
        print("-" * 60)


if __name__ == "__main__":
    main()