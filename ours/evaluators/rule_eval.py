from typing import Any, Sequence, List, Tuple, Dict
from ours.evaluators.base import BaseEvaluator
from ours.utils.registry import registry
import re
import numpy as np
import json

@registry.register_evaluator()
class CheXpertKeywordEvaluator(BaseEvaluator):
    evaluator_ids: List[str] = ['rule_chexpert_eval']

    def __init__(self, evaluator_id: str, metrics_cfg: Dict[str, Any], keyword_map: Dict[str, List[str]] = None) -> None:
        super().__init__(evaluator_id, metrics_cfg)
        # 定义每个CheXpert异常的关键词（可以自己扩展）
        self.keyword_map = keyword_map or {
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

    def process(self, preds: Sequence[Any], labels: Sequence[Any], extras: Sequence[Any]) -> Tuple[Sequence[Any], Sequence[Any]]:
        """
        preds: 模型生成的文本描述
        labels: 数据集标签（例如 [{'Cardiomegaly': 1, 'Edema': 0, ...}, ...])
        extras: 额外信息（可忽略）
        """
        label_names = list(self.keyword_map.keys())  # 保持固定顺序
        num_labels = len(label_names)

        y_pred = []
        y_true = []

        for pred, label_dict in zip(preds, labels):
            pred_vec = np.zeros(num_labels, dtype=int)
            true_vec = np.zeros(num_labels, dtype=int)

            # 1️⃣ 处理预测结果
            if isinstance(pred, str):
                pred_clean = re.sub(r"```json|```", "", pred).strip()
                try:
                    pred_dict = json.loads(pred_clean)
                    for i, disease in enumerate(label_names):
                        pred_vec[i] = int(pred_dict.get(disease, 0))
                except Exception as e:
                    print(f"⚠️ JSON parse error: {e} → default 0s")
            else:
                print(f"⚠️ Unexpected pred type: {type(pred)}")

            # 2️⃣ 处理真实标签
            for i, disease in enumerate(label_names):
                true_vec[i] = int(label_dict.get(disease, 0))

            y_pred.append(pred_vec)
            y_true.append(true_vec)

        processed_preds = np.array(y_pred)
        processed_labels = np.array(y_true)

        return processed_preds, processed_labels, extras