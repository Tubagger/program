import os
import argparse
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import json

# 读取CheXpert标签
def load_labels(csv_file):
    df = pd.read_csv(csv_file)
    return df

# 获取图像和标签信息
def get_images_and_labels(df, N, images_folder):
    selected_samples = df.head(N)
    images = []
    labels = []

    label_columns = [col for col in df.columns[5:] if col != "Support Devices"]  # ✅ 排除 Support Devices


    for _, row in selected_samples.iterrows():
        image_path = os.path.join(images_folder, row['Path']).replace('\\', '/')
        if not os.path.exists(image_path):
            print(f"⚠️ Warning: File not found: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")

        # 构建每个疾病标签的 0/1 字典
        disease_labels = {col: int(row[col]) if not pd.isna(row[col]) else 0 for col in label_columns}

        images.append(image)
        labels.append(disease_labels)

    return images, labels

# 保存图像和标签
def save_images_and_labels(images, labels, output_folder, json_output):
    os.makedirs(output_folder, exist_ok=True)
    saved_data = []

    for i, image in enumerate(images):
        filename = f"image_{i+1}.png"
        image_path = os.path.join(output_folder, filename)
        image.save(image_path)

        saved_data.append({
            "image_filename": filename,
            "labels": labels[i]  # 直接存字典
        })

    with open(json_output, 'w') as f:
        json.dump({"samples": saved_data}, f, indent=4)

    print(f"✅ Images saved to: {output_folder}")
    print(f"✅ Labels saved to: {json_output}")


# 显示图像和标签
def show_images_with_labels(images, labels):
    for i, image in enumerate(images):
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')
        label_text = ', '.join(labels[i])
        plt.title(label_text, fontsize=10, color='red')
        plt.show()

# 主程序
def main():
    parser = argparse.ArgumentParser(description="Process CheXpert dataset samples")
    parser.add_argument('--n', type=int, default=10, help='Number of images to process (default: 10)')
    parser.add_argument('--image_folder', type=str, default='../Dataset/', help='Path to the image root folder')
    parser.add_argument('--csv_file', type=str, default='../Dataset/CheXpert-v1.0-small/train.csv', help='Path to the CSV label file')
    parser.add_argument('--output_folder', type=str, default='./data/CheXpert-v1.0-small/images', help='Folder to save selected images')
    parser.add_argument('--json_output', type=str, default='./data/CheXpert-v1.0-small/labels.json', help='JSON file to save label info')
    args = parser.parse_args()

    # 1. 加载CheXpert标签文件
    df = load_labels(args.csv_file)

    # 2. 获取图像与对应标签
    images, labels = get_images_and_labels(df, args.n, args.image_folder)

    # 3. 保存图像与标签
    save_images_and_labels(images, labels, args.output_folder, args.json_output)

    # （可选）显示样本
    # show_images_with_labels(images, labels)

if __name__ == "__main__":
    main()
