import random

from torch.utils.data import DataLoader

from model_functions.diffusion import *
from model_functions.vae import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2024)

global_batch_size = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

input_dim = 1
dm_model = diff_STBlock(input_dim)
dm_model.to(device)

dm_optimizer = Adam(dm_model.parameters(), lr=1e-3)

pre_loss = 1e10

train_latents = np.load("../npy_files/train_latents.npy")


def add_gaussian_noise(latents, factor=0.08):
    """Adds Gaussian noise to the latent representations."""
    noise_level = factor * np.std(latents)
    return latents + np.random.normal(0, noise_level, latents.shape)


noisy_latents = [add_gaussian_noise(train_latents) for _ in range(9)]

augmented_latents = np.concatenate([
    train_latents,
    *noisy_latents
], axis=0)

train_latents = augmented_latents

train_latents = np.expand_dims(train_latents, 1).astype(np.float32)
train_spike_data = train_latents.transpose(0, 1, 3, 2)

dataloader = DataLoader(train_spike_data, batch_size=global_batch_size)

batch = next(iter(dataloader))

for epoch in range(n_epochs):
    total_loss = 0
    for step, batch in enumerate(dataloader):
        dm_optimizer.zero_grad()

        batch_size = batch.shape[0]
        batch = batch.to(device)

        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        loss = p_losses(dm_model, batch, t)
        total_loss += loss.item()

        loss.backward()
        dm_optimizer.step()

    print(f"total Loss of epoch {epoch} is {total_loss:.4f}")

    if total_loss < pre_loss:
        pre_loss = total_loss
        torch.save(dm_model.state_dict(), f'../model_checkpoints/source_diffusion_model.pth')
