import json
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# 读取样本数据
def load_samples(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['samples']

# 获取图像路径和标注信息
def get_images_and_labels(samples, N, images_folder):
    selected_samples = samples[:N]  # 按顺序选择 N 张图像
    images = []
    labels = []
    bounding_boxes = []
    
    for sample in selected_samples:
        image_path = os.path.join(images_folder, sample['filepath'])  # 图像路径
        ground_truth = sample['ground_truth']
        
        # 获取病灶标注信息
        detections = ground_truth['detections']
        label = detections[0]['label']  # 假设每张图像只有一个标注
        bounding_box = detections[0]['bounding_box']
        
        # 加载图像
        image = Image.open(image_path)
        
        # 保存图像和标注信息
        images.append(image)
        labels.append(label)
        bounding_boxes.append(bounding_box)
    
    return images, labels, bounding_boxes

# 保存图像并记录标注信息
def save_images_and_labels(images, bounding_boxes, labels, output_folder, json_output):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    saved_data = []

    for i, image in enumerate(images):
        # 图像保存路径
        image_filename = f"image_{i+1}.png"
        image_path = os.path.join(output_folder, image_filename)

        # 保存图像
        image.save(image_path)

        # 创建标注信息
        bounding_box = bounding_boxes[i]
        label = labels[i]

        # 将图像信息和标注添加到 JSON 数据中
        saved_data.append({
            "image_filename": image_filename,
            "label": label,
            "bounding_box": bounding_box
        })

    # 保存标注信息到新的 JSON 文件
    with open(json_output, 'w') as json_file:
        json.dump({"samples": saved_data}, json_file, indent=4)

    print(f"Images saved to: {output_folder}")
    print(f"Labels saved to: {json_output}")

# 显示图像及其标注
def show_images_with_bboxes(images, bounding_boxes, labels):
    for i, image in enumerate(images):
        # 获取边界框的坐标
        bbox = bounding_boxes[i]
        label = labels[i]
        
        # 将归一化坐标转换为像素值
        width, height = image.size
        x_min = int(bbox[0] * width)
        y_min = int(bbox[1] * height)
        w = int(bbox[2] * width)
        h = int(bbox[3] * height)
        
        # 绘制边界框
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        rect = plt.Rectangle((x_min, y_min), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min, label, color='white', fontsize=12, weight='bold', backgroundcolor='black')
        plt.show()

# 主程序
def main():
    # 设置命令行解析
    parser = argparse.ArgumentParser(description='Process and save images and labels.')
    
    # 设置 N 的默认值为 10
    parser.add_argument('--n', type=int, default=10, help='Number of images to process (default: 10)')
    
    # 设置其他路径
    parser.add_argument('--image_folder', type=str, default='../../Dataset/deeplesion-balanced-2k/', help='Path to the input image file(default: output_images)')
    parser.add_argument('--json_file', type=str, default='../../Dataset/deeplesion-balanced-2k/samples.json', help='Path to the input JSON file (default: samples.json)')
    parser.add_argument('--output_folder', type=str, default='../data/deeplesion-balanced-2k/images', help='Folder to save images (default: output_images)')
    parser.add_argument('--json_output', type=str, default='../data/deeplesion-balanced-2k/labels.json', help='Path to save the output JSON file (default: saved_labels.json)')
    
    args = parser.parse_args()  # 解析命令行参数
    
    # 1. 读取样本数据
    samples = load_samples(args.json_file)
    
    # 2. 获取图像和标签信息
    images, labels, bounding_boxes = get_images_and_labels(samples, args.n, args.image_folder)
     
    # 3. 保存图像和标注
    save_images_and_labels(images, bounding_boxes, labels, args.output_folder, args.json_output)

if __name__ == '__main__':
    main()
