import os
import sys
import argparse
import time
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add final_experiments and root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)

from common.reproducibility import set_seed, save_run_config
from common.dataset import JNU_IFM_Dataset, get_transforms
from common.training_utils import train_one_epoch, evaluate_loader, CheckpointManager, load_checkpoint
from resnet152.model import build_model
from resnet152.config import CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-152 on JNU-IFM Protocol A")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="Number of training epochs")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=CONFIG["batch_size"], help="Training batch size")
    parser.add_argument("--lr", type=float, default=CONFIG["lr"], help="Learning rate")
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=CONFIG["weight_decay"], help="Weight decay")
    parser.add_argument("--grad-accum-steps", "--grad_accum_steps", type=int, default=CONFIG["grad_accum_steps"], help="Gradient accumulation steps")
    parser.add_argument("--num-workers", "--num_workers", type=int, default=CONFIG["num_workers"], help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"], help="Random seed")
    parser.add_argument("--device", type=str, default=CONFIG["device"], help="Device: cuda or cpu")
    parser.add_argument("--pretrained", action="store_true", default=False, help="Use ImageNet pretrained weights (omit flag for scratch training)")
    parser.add_argument("--patience", type=int, default=CONFIG.get("patience", 7), help="Early stopping patience (epochs without validation Macro-F1 improvement; 0 to disable)")
    parser.add_argument("--train-csv", "--train_csv", type=str, default=CONFIG["train_csv"], help="Path to training CSV")
    parser.add_argument("--val-csv", "--val_csv", type=str, default=CONFIG["val_csv"], help="Path to development validation CSV")
    parser.add_argument("--output-dir", "--output_dir", type=str, default=None, help="Output directory (defaults to results/resnet152_<tier>)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    device_str = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    device = torch.device(device_str)
    print(f"[STARTUP] Initializing ResNet-152 training pipeline...")
    print(f"[INIT] Model: ResNet-152 | Device: {device} | Pretrained: {args.pretrained} | Seed: {args.seed}")

    if args.output_dir is None:
        tier_suffix = "pretrained" if args.pretrained else "scratch"
        output_dir = os.path.join(EXPERIMENTS_DIR, "results", f"{CONFIG['model_name']}_{tier_suffix}")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Record run config snapshot
    run_config = {
        "model_name": "resnet152",
        "init_mode": "pretrained" if args.pretrained else "random",
        "pretrained_weights": CONFIG.get("pretrained_weights") if args.pretrained else None,
        "pretrained_source": CONFIG.get("pretrained_source") if args.pretrained else None,
        "pretrained_url": CONFIG.get("pretrained_url") if args.pretrained else None,
        "early_stopping_patience": args.patience,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_accum_steps": args.grad_accum_steps,
        "seed": args.seed,
        "device": device_str,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "image_size": CONFIG["image_size"],
        "metric_for_best": CONFIG["metric_for_best"]
    }
    config_save_path = os.path.join(output_dir, "run_config.json")
    save_run_config(run_config, config_save_path)
    print(f"[CONFIG] Run configuration saved to: {config_save_path}")

    # Build datasets (Protocol A: train and val only)
    train_tf = get_transforms(image_size=CONFIG["image_size"], is_train=True)
    val_tf = get_transforms(image_size=CONFIG["image_size"], is_train=False)

    print(f"[DATASET] Loading training set: {args.train_csv}")
    train_dataset = JNU_IFM_Dataset(args.train_csv, transform=train_tf)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device_str == "cuda")
    )

    print(f"[DATASET] Loading development set: {args.val_csv}")
    val_dataset = JNU_IFM_Dataset(args.val_csv, transform=val_tf)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device_str == "cuda")
    )

    print(f"[DATASET] Train samples: {len(train_dataset)} | Dev samples: {len(val_dataset)}")

    # Instantiate model
    if args.pretrained:
        print(f"[PRETRAINED] Instantiating ResNet-152 with ImageNet-1K pretrained weights ({CONFIG.get('pretrained_weights')})...")
    model = build_model(num_classes=2, pretrained=args.pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    ckpt_manager = CheckpointManager(ckpt_dir, metric_name=CONFIG["metric_for_best"], mode='max')

    start_epoch = 1
    if args.resume:
        print(f"[RESUME] Resuming from checkpoint: {args.resume}")
        _, start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, device=device)
        start_epoch += 1

    # Initialize CSV training log
    log_csv_path = os.path.join(output_dir, "training_history.csv")
    csv_exists = os.path.exists(log_csv_path) and start_epoch > 1
    log_file = open(log_csv_path, "a" if csv_exists else "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if not csv_exists:
        log_writer.writerow([
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
            "val_macro_f1", "val_auc", "val_rec_class0", "val_spec_class0", "lr", "epoch_time_sec"
        ])

    print("\n[TRAINING START] Beginning training loop...")
    total_start_time = time.time()
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_accum_steps=args.grad_accum_steps
        )
        val_loss, val_metrics, _, _, _ = evaluate_loader(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - t0

        current_lr = optimizer.param_groups[0]['lr']
        macro_f1 = val_metrics["macro_f1"]
        auc = val_metrics["roc_auc"]
        rec0 = val_metrics["standard_plane_pos_label_0"]["sensitivity_recall"]
        spec0 = val_metrics["standard_plane_pos_label_0"]["specificity"]

        is_best = ckpt_manager.update(model, optimizer, scheduler, epoch, val_metrics, run_config)
        star = " *" if is_best else ""

        if is_best:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']*100:.1f}% "
            f"F1: {macro_f1:.4f} AUC: {auc:.4f}{star} "
            f"({epoch_time:.1f}s)"
        )

        log_writer.writerow([
            epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}",
            f"{val_metrics['accuracy']:.4f}", f"{macro_f1:.4f}", f"{auc:.4f}",
            f"{rec0:.4f}", f"{spec0:.4f}", f"{current_lr:.6f}", f"{epoch_time:.1f}"
        ])
        log_file.flush()

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"\n[EARLY STOPPING] Validation Macro-F1 did not improve for {args.patience} consecutive epochs. Early stopping triggered at epoch {epoch}.")
            break

    log_file.close()
    total_time = time.time() - total_start_time
    print(f"\n[COMPLETE] Training finished in {total_time/60:.2f} minutes.")
    print(f"[BEST MODEL] Best checkpoint saved to: {os.path.join(ckpt_dir, 'best_model.pth')}")

if __name__ == "__main__":
    main()
