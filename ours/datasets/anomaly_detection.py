from typing import Optional, Sequence, List
from ours.datasets.base import BaseDataset
from ours.methods.base import BaseMethod
from ours.utils.registry import registry
from ours import ImageTxtSample, _OutputType
import yaml
import json
import os

@registry.register_dataset()
class AnomalyData(BaseDataset):
    dataset_ids: Sequence[str] = [
        "anomaly-detection"
    ]
    
    dataset_config: Optional[str] = "./ours/configs/datasets/anomaly-detection.yaml"

    # 可选：定义关键词 map 用于评估报告或文本匹配
    keyword_map = {
        "Atelectasis": ["atelectasis", "collapse"],
        "Cardiomegaly": ["cardiomegaly", "enlarged heart"],
        "Consolidation": ["consolidation"],
        "Edema": ["edema", "fluid overload", "pulmonary edema"],
        "Pleural Effusion": ["effusion", "pleural effusion"],
        "Pneumonia": ["pneumonia", "infection"],
        "Pneumothorax": ["pneumothorax", "collapsed lung"],
        "Fracture": ["fracture", "broken rib"],
        "Support Devices": ["pacemaker", "tube", "catheter"],
    }

    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        """
        dataset_id: 数据集名称
        data_csv: CheXpert train/valid CSV 文件路径
        img_dir: 图像存放路径
        method_hook: 可选的预处理方法
        uncertainty_policy: 处理 -1 不确定标签的策略 ("U-Zero", "U-One", "U-Ignore")
        """
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.image_dir = self.config.get('image_dir')
        self.label_dir = self.config.get('annotation_file')
        self.nums = self.config.get('nums')
        assert os.path.exists(self.image_dir), f"❌ Image directory not found: {self.image_dir}"
        assert os.path.exists(self.label_dir), f"❌ Label file not found: {self.label_dir}"
        with open(self.label_dir, 'r', encoding='utf-8') as f:
            samples = json.load(f)['samples']

        assert self.nums <= len(samples), f"❌ num ({self.nums}) is larger than total samples ({len(samples)})."
        self.images = []
        self.labels = []
        # 只取前 nums 个样本
        samples = samples[:self.nums]

        # 定义 CheXpert 标签列（14 个）
        self.labels_columns = [
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
            "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
            "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
        ]

        for sample in samples:
            img = sample['image_filename']
            self.images.append(os.path.join(self.image_dir, img))
            self.labels.append(sample['labels'])  # 多标签

        prompt = (
            "Analyze this chest X-ray image. "
            "Answer '1' or '0' for each condition: "
            "No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, "
            "Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, "
            "Pleural Other, Fracture"
            # "Support Devices"  # 可选，根据需求是否保留
            "If any disease is present, set 'No Finding' to 0. Only one of 'No Finding' or other diseases can be 1."
            "Format your answer as a JSON with keys being the condition names and values being '1' or '0'."
        )
        dataset = []
        for img, labels in zip(self.images, self.labels):
            dataset.append(ImageTxtSample(
                image_path=img,
                text=prompt,        
                target=labels   # 多标签
            ))
            # print("img_dir",img)
        
        self.dataset = dataset


    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)
