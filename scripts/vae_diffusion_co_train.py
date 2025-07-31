import pickle
import random
import sys

import numpy as np
import scipy.signal as signal
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append('..')
from model_functions.diffusion import diff_STBlock, p_losses
from model_functions.vae import VAE_Model


def get_batches(x, batch_size):
    n_batches = len(x) // (batch_size)
    x = x[:n_batches * batch_size:]
    for n in range(0, x.shape[0], batch_size):
        x_batch = x[n:n + (batch_size)]
        yield x_batch


RAND_SEED = np.random.randint(10000)
print("RANDOM SEED: ", RAND_SEED)

# with open('datasets/Neural_Source.pkl', 'rb') as f:
#     train_data1 = pickle.load(f)['data']
with open('../datasets/source_data_array.pkl', 'rb') as f:
    train_data1 = pickle.load(f)

# train_trial_spikes1, train_trial_vel1 = train_data1['firing_rates'], train_data1['velocity']
train_trial_spikes1, train_trial_vel1 = train_data1['neural'], train_data1['vel']

# start_pos = 1
# end_pos = 1
#
# train_trial_spikes_tide1 = np.array(
#     [spike[start_pos:len_trial + start_pos, :num_neurons] for spike in train_trial_spikes1])
# print(np.shape(train_trial_spikes_tide1))
# train_trial_vel_tide1 = np.array([spike[start_pos:len_trial + start_pos, :] for spike in train_trial_vel1])
train_trial_spikes_tide1 = train_trial_spikes1
train_trial_vel_tide1 = train_trial_vel1
print(np.shape(train_trial_vel_tide1))

# bin_width = float(0.02) * 1000
bin_width = float(0.01) * 1000

train_trial_spikes_tide = train_trial_spikes_tide1
train_trial_vel_tide = train_trial_vel_tide1

# kern_sd_ms = 100
kern_sd_ms = float(0.01) * 1000 * 3
kern_sd = int(round(kern_sd_ms / bin_width))
window = signal.gaussian(kern_sd, kern_sd, sym=True)
window /= np.sum(window)
filt = lambda x: np.convolve(x, window, 'same')

train_trial_spikes_smoothed = np.apply_along_axis(filt, 1, train_trial_spikes_tide)

indices = np.arange(train_trial_spikes_tide.shape[0])
# np.random.seed(2023)
np.random.seed(RAND_SEED)
np.random.shuffle(indices)
train_len = round(len(indices) * 0.80)
real_train_trial_spikes_smed, val_trial_spikes_smed = train_trial_spikes_smoothed[indices[:train_len]], \
    train_trial_spikes_smoothed[indices[train_len:]]
real_train_trial_vel_tide, val_trial_vel_tide = train_trial_vel_tide[indices[:train_len]], train_trial_vel_tide[
    indices[train_len:]]

# n_epochs = 500
# read n_epochs from command line:
n_epochs = int(sys.argv[1])
batch_size = 32
ae_res_weight = 10
kld_weight = 1

n_batches = len(real_train_trial_spikes_smed) // batch_size
print(n_batches)

mse_criterion = nn.MSELoss()
poisson_criterion = nn.PoissonNLLLoss(log_input=False)

l_rate = 0.001

real_train_trial_spikes_stand = (real_train_trial_spikes_smed)
val_trial_spikes_stand = (val_trial_spikes_smed)

spike_train = Variable(torch.from_numpy(real_train_trial_spikes_stand)).float()
spike_val = Variable(torch.from_numpy(val_trial_spikes_stand)).float()

emg_val = Variable(torch.from_numpy(val_trial_vel_tide)).float()


def get_loss(model, spike, emg):
    re_sp_, vel_hat_, mu, log_var = model(spike, train_flag=True)
    ae_loss = poisson_criterion(re_sp_, spike)
    emg_loss = mse_criterion(vel_hat_, emg)
    kld_loss = torch.mean(0.5 * (- log_var + mu ** 2 + log_var.exp() - 1))
    total_loss = ae_res_weight * ae_loss + emg_loss + kld_weight * kld_loss
    return total_loss


# Training
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(21)
setup_seed(RAND_SEED)

pre_total_loss_ = 1e18

len_trial = train_trial_spikes_tide.shape[1]
num_neurons = train_trial_spikes_tide.shape[2]

model = VAE_Model(len_trial=len_trial, num_neurons=num_neurons)
optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

timesteps = 100

global_batch_size = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

input_dim = 1

dm_model = diff_STBlock(input_dim)
dm_model.to(device)

dm_optimizer = Adam(dm_model.parameters(), lr=1e-3)

pre_loss = 1e10

for epoch in range(n_epochs):
    spike_gen_obj = get_batches(real_train_trial_spikes_stand, batch_size)
    emg_gen_obj = get_batches(real_train_trial_vel_tide, batch_size)
    for ii in range(n_batches):
        optimizer.zero_grad()
        spike_batch = next(spike_gen_obj)
        emg_batch = next(emg_gen_obj)

        spike_batch = Variable(torch.from_numpy(spike_batch)).float()
        emg_batch = Variable(torch.from_numpy(emg_batch)).float()

        batch_loss = get_loss(model, spike_batch, emg_batch)

        batch_loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_total_loss = get_loss(model, spike_val, emg_val)

        _, _, train_latents, _ = model(spike_train, train_flag=False)

        if val_total_loss < pre_total_loss_:
            pre_total_loss_ = val_total_loss
            torch.save(model.state_dict(), '../model_checkpoints/source_vae_model.pth')
            np.save("../npy_files/train_latents.npy", train_latents)

    train_latents = np.expand_dims(train_latents, 1).astype(np.float32)
    train_spike_data = train_latents.transpose(0, 1, 3, 2)

    dataloader = DataLoader(train_spike_data, batch_size=global_batch_size)

    batch = next(iter(dataloader))

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

    print("epoch:", epoch, " " * 5, "loss:", total_loss)

    with torch.no_grad():
        if total_loss < pre_loss:
            pre_loss = total_loss
            torch.save(dm_model.state_dict(), '../model_checkpoints/source_diffusion_model.pth')
