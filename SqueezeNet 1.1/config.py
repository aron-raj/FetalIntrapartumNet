import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CONFIG = {
    "model_name": "squeezenet",
    "num_classes": 2,
    "pretrained": False,
    "pretrained_weights": "SqueezeNet1_1_Weights.IMAGENET1K_V1",
    "pretrained_source": "torchvision.models",
    "pretrained_url": "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
    "patience": 7,
    "image_size": (224, 224),
    "batch_size": 16,
    "epochs": 30,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "grad_accum_steps": 1,
    "num_workers": 0,
    "seed": 42,
    "device": "cuda",
    "metric_for_best": "macro_f1",
    "train_csv": os.path.join(PROJECT_ROOT, "jnu_ifm_binary_dataset", "splits", "train.csv"),
    "val_csv": os.path.join(PROJECT_ROOT, "jnu_ifm_binary_dataset", "splits", "validation.csv"),
    "test_csv": os.path.join(PROJECT_ROOT, "jnu_ifm_binary_dataset", "splits", "test.csv"),
    "output_dir": os.path.join(PROJECT_ROOT, "final_experiments", "results", "squeezenet"),
    "checkpoint_dir": os.path.join(PROJECT_ROOT, "final_experiments", "results", "squeezenet", "checkpoints"),
    "log_dir": os.path.join(PROJECT_ROOT, "final_experiments", "logs")
}
