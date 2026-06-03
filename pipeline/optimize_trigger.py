import argparse
import json
import os
import warnings
import csv
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.functional as F
import math

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import yaml
from types import SimpleNamespace

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
for path in (str(CURRENT_DIR), str(PARENT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from joint_loss import JointLoss
from models import get_model_adapter

warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="TypedStorage is deprecated",
    category=UserWarning,
)


def dict_to_namespace(d):
    """Recursively convert a dictionary to a nested SimpleNamespace."""
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return SimpleNamespace(**d)


def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {key: namespace_to_dict(value) for key, value in vars(obj).items()}
    return obj


def ensure_default_config(args):
    if not hasattr(args, "model"):
        args.model = SimpleNamespace(
            family="llava-1.5",
            model_path="liuhaotian/llava-v1.5-7b",
            device="cuda",
        )
    else:
        if not hasattr(args.model, "family"):
            args.model.family = "llava-1.5"
        if not hasattr(args.model, "model_path"):
            args.model.model_path = "liuhaotian/llava-v1.5-7b"
        if not hasattr(args.model, "device"):
            args.model.device = "cuda"

    return args


def validate_args(args):
    if not os.path.exists(args.dataset.dataset_split_path):
        raise FileNotFoundError(
            f"Dataset split file not found: {args.dataset.dataset_split_path}"
        )

    if args.model.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA, but no CUDA device is available.")

    if args.trigger_optim.lr_scheduler.type == "step" and not hasattr(
        args.trigger_optim.lr_scheduler, "step"
    ):
        raise ValueError("Step LR scheduler selected, but trigger_optim.lr_scheduler.step is missing.")

    if args.trigger_optim.optimizer.tpa_weight <= 0:
        raise ValueError("trigger_optim.optimizer.tpa_weight must be positive.")

    if (
        args.trigger_optim.optimizer.early_stop is not None
        and args.trigger_optim.optimizer.early_stop >= args.trigger_optim.train.epoch_num
    ):
        raise ValueError(
            "trigger_optim.optimizer.early_stop must be smaller than trigger_optim.train.epoch_num, or set to null."
        )




def parse_args():
    parser = argparse.ArgumentParser(description="Poisoning")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--poison_save_pth", type=str, required=True)

    cmd_args = parser.parse_args()

    with open(cmd_args.config, "r") as f:
        config = yaml.safe_load(f)

    args = dict_to_namespace(config)
    args.poison_save_pth = cmd_args.poison_save_pth
    args.config_path = cmd_args.config
    args = ensure_default_config(args)
    validate_args(args)
    return args


def load_image_tensors(args, img_size, valid=False):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    with open(args.dataset.dataset_split_path) as f:
        dataset_split = json.load(f)
    dataset_info = dataset_split[args.dataset.image_set]
    clean_image_root_path = dataset_info["root_path"]
    image_fn_list = dataset_info["trigger_optimization"]

    train_num = args.dataset.split_sample_num.train
    valid_num = args.dataset.split_sample_num.valid

    if train_num + valid_num > len(image_fn_list):
        raise ValueError(
            f"Requested train={train_num} and valid={valid_num}, but only "
            f"{len(image_fn_list)} trigger_optimization images are available."
        )

    start_idx = train_num if valid else 0
    num_images = valid_num if valid else train_num

    preview_list = []
    clean_image_list = []

    def resolve_image_path(image_root, image_name):
        candidate_names = [image_name]
        if not Path(image_name).suffix:
            candidate_names.extend([f"{image_name}.jpg", f"{image_name}.png", f"{image_name}.jpeg"])

        for candidate_name in candidate_names:
            candidate_path = os.path.join(image_root, candidate_name)
            if os.path.exists(candidate_path):
                return candidate_path

        raise FileNotFoundError(f"Unable to resolve image path for {image_name} under {image_root}")

    for i in range(num_images):
        image_fn = image_fn_list[start_idx + i]
        if len(preview_list) < 5:
            preview_list.append(image_fn)
            
        p = resolve_image_path(clean_image_root_path, image_fn)
        image_tensor = transform(Image.open(p).convert('RGB')).unsqueeze(0)
        clean_image_list.append(image_tensor)

    clean_image_list = torch.cat(clean_image_list, axis=0)
  

    print(f"=== load_image_tensors (valid={valid}) ===")
    print(f"   clean_image_list.shape:   {clean_image_list.shape}")
    print(f"   [0:5]:({preview_list})")

    return clean_image_list


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        super().__init__()
        self.images = images

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return (self.images[index], index)


def manual_seed(seed: int):
    from torch.backends import cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.model.device)

    os.makedirs(os.path.join(args.poison_save_pth, "patch"), exist_ok=False)
    print(f"Poison images will be saved to {args.poison_save_pth}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(args.poison_save_pth, f"training_log_{timestamp}.csv")
    csv_header = ["epoch", "cur_val_loss", "cur_val_loss_tpa", "cur_val_loss_psp"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    print("=" * 24)
    print("Configuration: \n")
    print(yaml.dump(namespace_to_dict(args), default_flow_style=False, sort_keys=False))
    print("=" * 24)

    manual_seed(args.trigger_optim.train.seed)
    model_adapter = get_model_adapter(args.model.family)
    model_spec = model_adapter.spec
    image_encoder, _image_processor = model_adapter.load_image_encoder(
        getattr(args.model, "model_path", None),
        device=args.model.device,
    )

    img_size = model_spec.image_size
    dataset_train = ImageDataset(load_image_tensors(args, img_size))
    dataset_valid = ImageDataset(load_image_tensors(args, img_size, valid=True))

    eps = args.trigger.eps / 255.0
    patch = torch.rand(
        [args.trigger.num, 3, img_size, img_size],
        dtype=torch.float32,
        device=device,
    ) * 2 * eps - eps
    patch.requires_grad_(True)

    joint_loss = JointLoss(
        num_classes=args.trigger.num + 1,
        feat_dim=model_spec.feature_dim,
        device=args.model.device,
    )

    optimizer_psp = optim.SGD([patch], lr=args.trigger_optim.optimizer.lr_psp / 255.0)
    optimizer_tpa = optim.SGD(joint_loss.parameters(), lr=args.trigger_optim.optimizer.lr_tpa)

    def lr_lambda_step(epoch):
        ratio = 1.0
        for milestone in args.trigger_optim.lr_scheduler.step.milestones:
            if epoch >= milestone:
                ratio *= args.trigger_optim.lr_scheduler.step.gamma
        return ratio

    def lr_lambda_cosine(epoch):
        warmup_epochs = args.trigger_optim.lr_scheduler.cosine.warmup_epochs
        T_max = args.trigger_optim.train.epoch_num
        eta_min = args.trigger_optim.lr_scheduler.cosine.eta_min

        if epoch < warmup_epochs:
            ratio = epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (T_max - warmup_epochs)
            ratio = eta_min + (1 - eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
        return ratio

    if args.trigger_optim.lr_scheduler.type == "step":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_psp, lr_lambda=lr_lambda_step)
        scheduler_center = torch.optim.lr_scheduler.LambdaLR(optimizer_tpa, lr_lambda=lr_lambda_step)
    elif args.trigger_optim.lr_scheduler.type == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_psp, lr_lambda=lr_lambda_cosine)
        scheduler_center = torch.optim.lr_scheduler.LambdaLR(optimizer_tpa, lr_lambda=lr_lambda_cosine)
    else:
        raise NotImplementedError

    def apply_patch_to_image(clean_img, patch):
        return (clean_img.detach() + patch).clamp(0, 1)

    patch_save_path = os.path.join(args.poison_save_pth, "patch")
    best_val_loss = 1e8

    dataloader_base_val = DataLoader(dataset_valid, batch_size=args.trigger_optim.train.batch_size, shuffle=True)
    dataloader_base = DataLoader(dataset_train, batch_size=args.trigger_optim.train.batch_size, shuffle=True)

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    for epoch_index in range(args.trigger_optim.train.epoch_num):
        if (
            args.trigger_optim.optimizer.early_stop is not None
            and epoch_index > args.trigger_optim.optimizer.early_stop
        ):
            print(f"Early stop at epoch #{args.trigger_optim.optimizer.early_stop}")
            break

        print(f"==== epoch {epoch_index:04d} ====")

        current_lr_psp = optimizer_psp.param_groups[0]["lr"]
        print(f"    current learning rate for psp * 255 = {current_lr_psp * 255.0}")

        current_lr_tpa = optimizer_tpa.param_groups[0]["lr"]
        print(f"    current learning rate for tpa       = {current_lr_tpa}")

        optimizer_psp.zero_grad()
        optimizer_tpa.zero_grad()

        for i, (clean_image, index_base) in enumerate(dataloader_base):
            del index_base
            clean_image = clean_image.to(device)
            all_class_patches = []

            with torch.no_grad():
                clean_image_emb = image_encoder(normalize(clean_image))
                all_class_patches.append(clean_image_emb)

            for patch_index in range(args.trigger.num):
                poison_image = apply_patch_to_image(clean_image, patch[patch_index].unsqueeze(0))
                poison_image_emb = image_encoder(normalize(poison_image))
                all_class_patches.append(poison_image_emb)

            all_class_patches = torch.cat(all_class_patches, dim=0)
            all_class_patches = all_class_patches.view(all_class_patches.shape[0], -1)
            batch_size = clean_image.shape[0]
            gt_label = torch.arange(args.trigger.num + 1, device=device).repeat_interleave(batch_size)
            loss_tpa, loss_psp, _ = joint_loss(all_class_patches, gt_label)
            loss = loss_psp + args.trigger_optim.optimizer.tpa_weight * loss_tpa

            loss.backward()
            if patch.grad is None:
                raise RuntimeError("Patch gradient is missing after backward().")
            patch.grad.sign_()

            optimizer_psp.step()
            optimizer_psp.zero_grad()

            for param in joint_loss.parameters():
                if param.grad is not None:
                    param.grad.mul_(1.0 / args.trigger_optim.optimizer.tpa_weight)

            optimizer_tpa.step()
            optimizer_tpa.zero_grad()
            with torch.no_grad():
                patch.clamp_(-eps, eps)

        scheduler.step()
        scheduler_center.step()
        loss_sum_on_batch = 0.0
        loss_tpa_sum_on_batch = 0.0
        loss_psp_sum_on_batch = 0.0

        with torch.no_grad():
            logits_list = [[] for _ in range(args.trigger.num + 1)]

            for i, (clean_image, index_base) in enumerate(dataloader_base_val):
                del i, index_base
                clean_image = clean_image.to(device)
                all_class_patches = []

                clean_image_emb = image_encoder(normalize(clean_image))
                all_class_patches.append(clean_image_emb)

                for patch_index in range(args.trigger.num):
                    poison_image = apply_patch_to_image(clean_image, patch[patch_index].unsqueeze(0))
                    poison_image_emb = image_encoder(normalize(poison_image))
                    all_class_patches.append(poison_image_emb)

                all_class_patches = torch.cat(all_class_patches, dim=0)
                all_class_patches = all_class_patches.view(all_class_patches.shape[0], -1)
                batch_size = clean_image.shape[0]
                gt_label = torch.arange(args.trigger.num + 1, device=device).repeat_interleave(batch_size)

                loss_tpa, loss_psp, logits = joint_loss(all_class_patches, gt_label)
                loss = loss_psp + args.trigger_optim.optimizer.tpa_weight * loss_tpa
                logits_split = torch.split(logits, batch_size)
                for class_idx in range(args.trigger.num + 1):
                    logits_list[class_idx].append(logits_split[class_idx])

                loss_sum_on_batch += loss.item()
                loss_tpa_sum_on_batch += loss_tpa.item()
                loss_psp_sum_on_batch += loss_psp.item()

            logits_list = [torch.cat(logits, dim=0) for logits in logits_list]
            probs_list = [F.softmax(logits, dim=1).mean(dim=0) for logits in logits_list]

        cur_val_loss = loss_sum_on_batch / len(dataloader_base_val)
        cur_val_loss_tpa = loss_tpa_sum_on_batch / len(dataloader_base_val)
        cur_val_loss_psp = loss_psp_sum_on_batch / len(dataloader_base_val)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch_index,
                cur_val_loss,
                cur_val_loss_tpa, 
                cur_val_loss_psp,
            ])
        
        print(f"    cur_val_loss        = {cur_val_loss}")
        print(f"    cur_val_loss_tpa    = {cur_val_loss_tpa}")
        print(f"    cur_val_loss_psp    = {cur_val_loss_psp}")
        print(f"    probs each patch    = {probs_list}")

        should_save_patch = (
            cur_val_loss < best_val_loss
            or epoch_index >= int(args.trigger_optim.train.epoch_num * 0.90)
            or epoch_index % 25 == 0
        )
        if should_save_patch:
            if cur_val_loss < best_val_loss:
                print("    [new best]")
                best_val_loss = cur_val_loss

            for patch_idx in range(args.trigger.num):
                patch_dir = os.path.join(patch_save_path, f"patch_{patch_idx}")
                if not os.path.exists(patch_dir):
                    os.makedirs(patch_dir)

                current_patch = patch[patch_idx].detach().clone().cpu()
                patch_normalized = (current_patch + eps) / (2 * eps)
                to_pil = transforms.ToPILImage()
                patch_image = to_pil(patch_normalized)
                patch_image.save(os.path.join(patch_dir, f"{epoch_index:03d}.png"))
                print(f"    ** patch[{patch_idx}] range: ({patch[patch_idx].min():.3f}, {patch[patch_idx].max():.3f})")

        print(f"=" * 48 + "(valid)")
