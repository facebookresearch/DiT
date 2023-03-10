# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Sample new images from a pre-trained DiT.
"""
import random

import torch
import argparse
import gradio as gr
import torchvision

from torchvision.utils import make_grid
from diffusers.models import AutoencoderKL

from modules.diffusion import create_diffusion
from download import find_model
from modules.dit_builder import DiT_models

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def sample(prompt, cfg_scale, num_sampling_steps, seed):
    # Setup PyTorch:
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    model.to(device)
    model.eval()
    diffusion = create_diffusion(str(num_sampling_steps))

    bsize = 1
    z = torch.randn(bsize, 4, latent_size, latent_size, device=device)
    y = model.encode(prompt).squeeze(1).repeat(bsize, 1, 1).to(device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = model.encode("").squeeze(1).repeat(bsize, 1, 1).to(device)  # negative
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

    # OOMs :cry:
    model.cpu()
    torch.cuda.empty_cache()

    vae.to(device)
    samples = vae.decode(samples / 0.18215).sample
    vae.cpu()

    # Save and display images:
    samples = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
    samples = torchvision.transforms.ToPILImage()(samples)
    return samples
    # save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT_Clipped")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
    ).to(device)
    if args.ckpt:
        print(f"Loading {args.ckpt}")
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        if 'pytorch-lightning_version' in state_dict.keys():
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").cpu()

    demo = gr.Interface(
        fn=sample,
        inputs=[
            gr.Text(label="Text Prompt", value="an apple"),
            gr.Slider(minimum=1, maximum=20, value=4, step=0.1, label="Cfg scale"),
            gr.Slider(minimum=5, maximum=1000, value=50, step=1, label="Sampling steps"),
            gr.Slider(minimum=1, maximum=9223372036854775807, value=5782510030869745000, step=1,
                      label="Seed"),
        ],
        outputs=[
            gr.Image()
        ]
    )
    demo.launch()
