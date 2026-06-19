# experiments/create_super_hard_unknown_test.py
import json
from pathlib import Path

INPUT_PATH = "/Users/anastasiya/Documents/AITourGuide/data/processed/dataset_v1/val.json"
OUTPUT_PATH = "/Users/anastasiya/Documents/AITourGuide/data/processed/dataset_v1/val_super_hard_unknown.json"
MIN_SCORE = 0.85  # Только супер-hard negatives
MIN_HARD_NEGATIVES = 3
MAX_CANDIDATES = 15

def create_super_hard_unknown_dataset():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    super_hard_samples = []
    
    for sample in test_data:
        target_idx = sample.get("target_idx", -1)
        if target_idx == -1:
            continue
        
        candidates = sample.get("candidates", [])
        
        # Берем только candidates с retrieval_score >= MIN_SCORE
        super_hard_negatives = [
            cand for idx, cand in enumerate(candidates)
            if idx != target_idx 
            and cand.get("retrieval_score", 0.0) >= MIN_SCORE
        ]
        
        if len(super_hard_negatives) >= MIN_HARD_NEGATIVES:
            super_hard_sample = {
                "query_image": sample["query_image"],
                "candidates": super_hard_negatives[:MAX_CANDIDATES],
                "target_idx": -1,
                "meta": {
                    **sample.get("meta", {}),
                    "is_super_hard_unknown": True,
                    "min_score": MIN_SCORE
                }
            }
            super_hard_samples.append(super_hard_sample)
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(super_hard_samples, f, indent=2, ensure_ascii=False)
    
    print(f"Создано {len(super_hard_samples)} super-hard unknown сэмплов (score >= {MIN_SCORE})")
    print(f"Сохранено в: {OUTPUT_PATH}")

if __name__ == "__main__":
    create_super_hard_unknown_dataset()