import os
import typing
import subprocess
import torch
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_XL_2
from PIL import Image

torch.set_grad_enabled(False)

from cog import BasePredictor, Path, Input

os.environ['PYTHONPATH'] = '.'
device = "cuda"

from Replicate_demo.ImageNet_classnames import IMAGENET_1K_CLASSES
choices = sorted(list(IMAGENET_1K_CLASSES.keys()))

class Predictor(BasePredictor):
    def setup(self):
        subprocess.run(["mkdir", "/root/.cache/huggingface"])
        subprocess.run(["cp", "-r", "diffusers", "/root/.cache/huggingface/"])

        # Load model:
        self.latent_size_256 = 256 // 8
        self.model_256 = DiT_XL_2(input_size=self.latent_size_256).to(device)
        self.model_256.load_state_dict(find_model(f"DiT-XL-2-256x256.pt"))
        self.model_256.eval()  # important!

        self.latent_size_512 = 512 // 8
        self.model_512 = DiT_XL_2(input_size=self.latent_size_512).to(device)
        self.model_512.load_state_dict(find_model(f"DiT-XL-2-512x512.pt"))
        self.model_512.eval()  # important!

        self.vae_mse = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        self.vae_ema = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

    def predict(
            self,
            DiT_resolution: str = Input(description="Output image size", default="256x256", choices=["256x256", "512x512"]),
            VAE_Decoder: str = Input(description="decoder type", default='sd-vae-ft-mse', choices=['sd-vae-ft-mse', 'sd-vae-ft-ema']),
            num_sampling_steps: int = Input(description="Number of denoising steps", ge=0, le=1000, default=250),
            cfg_scale: float = Input(description="", ge=1, le=10., default=4),
            class_name: str = Input(description="Which ImageNet class to generate", choices=choices, default='centipede'),
            num_outputs: int = Input(description="How many outputs", default=4),
    ) -> typing.List[Path]:
        if str(DiT_resolution) == "256x256":
            latent_size = self.latent_size_256
            model = self.model_256
        else:
            latent_size = self.latent_size_512
            model = self.model_512

        if str(VAE_Decoder) == 'sd-vae-ft-mse':
            vae = self.vae_mse
        else:
            vae = self.vae_ema

        # Create diffusion object:
        diffusion = create_diffusion(str(num_sampling_steps))

        # Create sampling noise:
        n = int(num_outputs)
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.tensor([IMAGENET_1K_CLASSES[str(class_name)]] * n, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=float(cfg_scale))

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=True, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        outputs = []
        for i in range(len(samples)):
            path = f"output-{i}.png"
            save_image(samples[i], path, normalize=True, value_range=(-1, 1))
            outputs.append(Path(path))
        return outputs

