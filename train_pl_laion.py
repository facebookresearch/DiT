import argparse

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from datasets import load_dataset

from modules.dit_builder import DiT_models
from modules.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from modules.training_utils import *


def train_pl(args):
    print("Starting training..")
    laion_dataset = load_dataset("ChristophSchuhmann/improved_aesthetics_6.5plus")["train"]

    device = torch.device(0)

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
    )
    # ema = deepcopy(model).cpu()
    # requires_grad(ema, False)

    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    # training only
    model.diffusion = diffusion
    # model.ema = ema
    model.vae = vae

    loader_train = DataLoader(
        laion_dataset,
        batch_size=args.global_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    # update_ema(ema, model.module, decay=0)
    model.train().to(device)
    # ema.eval()

    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        auto_lr_find=True,
        enable_checkpointing=True,
        detect_anomaly=True,
        log_every_n_steps=50,
        accelerator='gpu',
        devices=1,
        max_epochs=args.epochs,
        precision=16
    )
    trainer.fit(model, loader_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT_Clipped")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parsed_args = parser.parse_args()
    train_pl(parsed_args)
