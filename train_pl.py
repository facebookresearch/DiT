import argparse
import random

import torchvision
import torch

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from download import find_model
from modules.dit_builder import DiT_models
from modules.diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from modules.image_cap_dataset import HuggingfaceImageDataset, DummyDataset, HuggingfaceImageNetDataset
from modules.training_utils import center_crop_arr


def train_pl(args):
    print("Starting training..")
    secondary_device = torch.device("cpu")

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        compile_components=False
    )

    # dataset1 = HuggingfaceImageNetDataset(args.token, res=args.image_size)
    dataset = torchvision.datasets.CocoCaptions(args.coco_dataset_path + "/train2017/",
                                                args.coco_dataset_path + "/annotations_train/captions_train2017.json",
                                                transform=transform,
                                                target_transform=lambda x: random.choice(x).replace("\n", "").lower())
    # dataset3 = HuggingfaceImageDataset(args.hf_dataset_name, args.token, res=args.image_size)
    # dataset = DummyDataset(res=args.image_size)

    # dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])

    state_dict = find_model("pretrained_models/last.ckpt")
    if 'pytorch-lightning_version' in state_dict.keys():
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)

    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(secondary_device)
    # training only
    model.diffusion = diffusion
    model.vae = vae
    model.secondary_device = secondary_device

    grad_batch_accum = 6
    model_ckpt = ModelCheckpoint(dirpath="ckpts/", monitor="train_loss", save_top_k=2, save_last=True,
                                 every_n_train_steps=3_000 // grad_batch_accum)

    loader_train = DataLoader(
        dataset,
        batch_size=args.global_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # update_ema(ema, model.module, decay=0)
    model.train()
    # ema.eval()

    torch.set_float32_matmul_precision("high")
    trainer = L.Trainer(
        enable_checkpointing=True,
        detect_anomaly=False,
        log_every_n_steps=50 // grad_batch_accum,
        accelerator="auto",
        devices="auto",
        max_epochs=args.epochs,
        precision="16-mixed" if args.precision == "fp16" else "32-true",
        callbacks=[model_ckpt],
        # StochasticWeightAveraging(swa_lrs=1e-2)],
        accumulate_grad_batches=grad_batch_accum,
        # move_metrics_to_cpu=True
    )

    trainer.fit(model, loader_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_dataset_name", type=str, default="facebook/winoground", required=False)
    parser.add_argument("--token", type=str, required=False)

    parser.add_argument("--coco_dataset_path", type=str, required=False)

    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT_Clipped")
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parsed_args = parser.parse_args()

    assert (parsed_args.hf_dataset_name is not None and parsed_args.token is not None) or (
        parsed_args.coco_dataset_path) is not None, "Either hf token and dataset name or coco dataset path is required"

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, parsed_args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    train_pl(parsed_args)
