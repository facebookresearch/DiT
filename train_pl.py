import argparse

import pytorch_lightning as pl
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from modules.dit_builder import DiT_models
from modules.diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from modules.image_cap_dataset import ImageCaptionDataset
from modules.training_utils import *


def train_pl(args):
    global model
    print("Starting training..")
    device = torch.device(0)

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
    )
    dataset = ImageCaptionDataset(args.hf_dataset_name, args.token, res=args.image_size)

    # del model.encoder

    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").cpu()
    # training only
    model.diffusion = diffusion
    model.vae = vae
    model_ckpt = ModelCheckpoint(dirpath="ckpts/", monitor=None, save_top_k=-1, save_last=True, every_n_train_steps=1000
                                 )

    loader_train = DataLoader(
        dataset,
        batch_size=args.global_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # update_ema(ema, model.module, decay=0)
    model.train().to(device)
    # ema.eval()

    torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        auto_lr_find=True,
        enable_checkpointing=True,
        detect_anomaly=True,
        log_every_n_steps=50,
        accelerator='gpu',
        devices=1,
        max_epochs=args.epochs,
        precision=16 if args.precision == "fp16" else 32,
        callbacks=[model_ckpt]
    )

    trainer.fit(model, loader_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_dataset_name", type=str, default="facebook/winoground")
    parser.add_argument("--token", type=str)

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

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, parsed_args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    train_pl(parsed_args)
