import torch
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_XL_2


# Setup PyTorch:
torch.manual_seed(0)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
num_sampling_steps = 250
cfg_scale = 4.0

# Load model:
model = DiT_XL_2().to(device)
state_dict = find_model("DiT-XL-2-256x256.pt")
model.load_state_dict(state_dict)
model.eval()
diffusion = create_diffusion(str(num_sampling_steps))
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

# Create sampling noise:
n = len(class_labels)
z = torch.randn(n, 4, 32, 32, device=device)
y = torch.tensor(class_labels, device=device)

# Setup classifier-free guidance:
z = torch.cat([z, z], 0)
y_null = torch.tensor([1000] * n, device=device)
y = torch.cat([y, y_null], 0)
model_kwargs = dict(y=y, cfg_scale=cfg_scale)

# Sample images:
samples = diffusion.p_sample_loop(
    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
)
samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
samples = vae.decode(samples / 0.18215).sample

# Save and display images:
save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
