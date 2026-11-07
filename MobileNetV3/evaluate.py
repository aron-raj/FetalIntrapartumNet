import os
import sys
import argparse
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)

from common.dataset import JNU_IFM_Dataset, get_transforms, DEFAULT_TEST_CSV
from common.metrics import compute_all_metrics, format_metrics_table
from common.complexity import profile_model_full
from mobilenetv3.model import build_model
from mobilenetv3.config import CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV3-Large on Locked Test Split")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--test-csv", "--test_csv", type=str, default=DEFAULT_TEST_CSV, help="Locked test split CSV")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--output-dir", "--output_dir", type=str, default=None, help="Output directory (defaults to checkpoint parent directory)")
    return parser.parse_args()

def main():
    args = parse_args()
    device_str = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    device = torch.device(device_str)

    if args.output_dir is None:
        ckpt_parent = os.path.dirname(os.path.abspath(args.checkpoint))
        output_dir = os.path.dirname(ckpt_parent) if os.path.basename(ckpt_parent) == "checkpoints" else ckpt_parent
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f" EVALUATING MOBILENETV3-LARGE ON LOCKED TEST SET")
    print(f" Checkpoint: {args.checkpoint}")
    print(f" Test Split: {args.test_csv}")
    print(f" Device    : {device}")
    print(f"=======================================================")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    model = build_model(num_classes=2, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    print("\n[PROFILING] Measuring parameter count, FLOPs, and inference latency...")
    complexity = profile_model_full(model, input_size=(3, 224, 224), device=device_str)

    test_tf = get_transforms(image_size=(224, 224), is_train=False)
    test_dataset = JNU_IFM_Dataset(args.test_csv, transform=test_tf, return_meta=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"[DATASET] Loaded {len(test_dataset)} locked test images.")

    all_preds = []
    all_labels = []
    all_probs = []
    per_image_rows = []

    with torch.no_grad():
        for images, labels, paths, video_ids in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            p_cpu = preds.cpu().numpy().tolist()
            l_cpu = labels.cpu().numpy().tolist()
            prob_cpu = probs.cpu().numpy().tolist()

            all_preds.extend(p_cpu)
            all_labels.extend(l_cpu)
            all_probs.extend(prob_cpu)

            for i in range(len(paths)):
                per_image_rows.append({
                    "image_path": paths[i],
                    "video_id": video_ids[i],
                    "true_label": l_cpu[i],
                    "pred_label": p_cpu[i],
                    "prob_class_0_standard": round(float(prob_cpu[i][0]), 6),
                    "prob_class_1_nonstandard": round(float(prob_cpu[i][1]), 6),
                    "is_correct": int(l_cpu[i] == p_cpu[i])
                })

    metrics = compute_all_metrics(all_labels, all_preds, all_probs)
    table_str = format_metrics_table(metrics, model_name="MobileNetV3-Large (Locked Test Set)")
    print("\n" + table_str)

    print(f"\nComplexity Metrics:")
    print(f" Total Parameters  : {complexity['total_parameters']:,} ({complexity['total_parameters_million']}M)")
    print(f" GFLOPs            : {complexity['gflops']}")
    print(f" Latency (avg)     : {complexity['avg_latency_ms']} ms")
    print(f" Throughput        : {complexity['throughput_images_per_sec']} img/s")

    eval_dir = os.path.join(output_dir, "eval_locked_test")
    os.makedirs(eval_dir, exist_ok=True)

    metrics_out = {
        "metrics": metrics,
        "complexity": complexity,
        "checkpoint": args.checkpoint,
        "test_csv": args.test_csv
    }

    with open(os.path.join(eval_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=4)

    with open(os.path.join(eval_dir, "evaluation_summary.txt"), "w", encoding="utf-8") as f:
        f.write(table_str + "\n\nComplexity:\n" + json.dumps(complexity, indent=4))

    pd.DataFrame(per_image_rows).to_csv(os.path.join(eval_dir, "per_image_predictions.csv"), index=False)
    print(f"\n[SAVED] All locked test evaluation artifacts saved to: {eval_dir}")

if __name__ == "__main__":
    main()
