import argparse
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils_scripts.data_prepare import load_tensors_by_index, cut_trials

sys.path.append('..')
from model_functions.diffusion import diff_STBlock, q_sample
from model_functions.mla_model import VAE_MLA_Model
from model_functions.vae_readout import VAE_Readout_Model
from utils_scripts.utils_torch import setup_seed, SpikeDataset, logger_performance
from utils_scripts.utils_torch import vel_cal

parser = argparse.ArgumentParser(description="Set hyperparameters from command line")

parser.add_argument("--learning_rate", type=float, default=3e-3, help="Learning rate for optimizer")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--appro_alpha", type=float, default=0.0, help="Approximator alpha parameter")
parser.add_argument("--ot_weight", type=float, default=0.8, help="Weight for optimal transport loss")
parser.add_argument("--epochs", type=int, default=400, help="Alternative epoch count (possible typo in config)")
parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducibility")
parser.add_argument("--dataset_name", type=str, default='erdiff_synthetic_npz', help="Dataset name")
parser.add_argument("--trial_len", type=int, default=25, help="Trial length")
parser.add_argument("--source_day", type=int, default=0, help="Source day")
parser.add_argument("--target_day", type=int, default=1, help="Target day")

args = parser.parse_args()

config = vars(args)

pre_total_loss_ = 1e8
best_metric = -1000
l_rate = config["learning_rate"]
batch_size = config["batch_size"]
ot_weight = config["ot_weight"]
appro_alpha = config["appro_alpha"]
epochs = config["epochs"]
n_epochs = config["epochs"]
random_seed = config["seed"]
dataset_name = config["dataset_name"]
trial_len = config["trial_len"]
source_day = config["source_day"]
target_day = config["target_day"]

print("Config Data:", config)

train_spikes_concat, train_vel_concat = load_tensors_by_index(source_day, dataset_name)
train_trial_spikes_smoothed, train_trial_vel_tide = cut_trials(train_spikes_concat, train_vel_concat, trial_len)

test_spikes_concat, test_vel_concat = load_tensors_by_index(target_day, dataset_name)
test_trial_spikes_smoothed, test_trial_vel_tide = cut_trials(test_spikes_concat, test_vel_concat, trial_len)

print(f"dataset:: original shapes: {train_spikes_concat.shape}, {train_vel_concat.shape}     "
      f"cut shapes: {train_trial_spikes_smoothed.shape}, {train_trial_vel_tide.shape}")

timesteps = 100
channels = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

input_dim = 1

diff_model = diff_STBlock(input_dim).to(device)

diff_model_dict = torch.load('../model_checkpoints/source_diffusion_model.pth', map_location=torch.device('cpu'),
                             weights_only=True)
diff_model.load_state_dict(diff_model_dict)

for k, v in diff_model.named_parameters():
    v.requires_grad = False

setup_seed(config["seed"])

vanilla_model_dict = torch.load('../model_checkpoints/source_vae_model.pth', weights_only=True,
                                map_location=torch.device('cpu'))

len_trial = train_trial_spikes_smoothed.shape[1]
num_neurons_s = train_trial_spikes_smoothed.shape[2]
num_neurons_t = test_trial_spikes_smoothed.shape[2]
vel_dim = train_trial_vel_tide.shape[2]

MLA_model = VAE_MLA_Model(len_trial, num_neurons_s, num_neurons_t, vel_dim).to(device)
MLA_dict_keys = MLA_model.state_dict().keys()
vanilla_model_dict_keys = vanilla_model_dict.keys()

MLA_dict_new = MLA_model.state_dict().copy()

for key in vanilla_model_dict_keys:
    MLA_dict_new[key] = vanilla_model_dict[key]

MLA_model.load_state_dict(MLA_dict_new)

optimizer = torch.optim.SGD(MLA_model.parameters(), lr=l_rate)
criterion = nn.MSELoss()
poisson_criterion = nn.PoissonNLLLoss(log_input=False)

# Freeze the other parameters
for param in MLA_model.parameters():
    param.requires_grad = False

MLA_model.align_layer.weight.requires_grad = True
MLA_model.low_d_readin_t_2.weight.requires_grad = True
MLA_model.low_d_readin_t_2.bias.requires_grad = True

spike_day_0 = Variable(torch.from_numpy(train_trial_spikes_smoothed)).float().to(device)
spike_day_k = Variable(torch.from_numpy(test_trial_spikes_smoothed)).float().to(device)
spike_dataset = SpikeDataset(spike_day_0, spike_day_k)

print(f'spike_day_0 shape: {spike_day_0.shape}, spike_day_k shape: {spike_day_k.shape}')

dataloader = DataLoader(spike_dataset, batch_size=batch_size, shuffle=False)

num_y_test = test_trial_spikes_smoothed.shape[0]
q_test = Variable(torch.from_numpy(np.full((num_y_test, 1), 1 / num_y_test))).float().to(device)

timestamp = datetime.now().strftime("%m%d_%H%M")
exp_name = f'ERDiff_MLA_{timestamp}'

# Maximum Likelihood Alignment
for epoch in range(epochs):

    optimizer.zero_grad()

    for batch in dataloader:
        batch_day_0 = batch[0].to(device)
        batch_day_k = batch[1].to(device)

        num_x, num_y = batch_day_0.shape[0], batch_day_k.shape[0]

        p = Variable(torch.from_numpy(np.full((num_x, 1), 1 / num_x))).float().to(device)
        q = Variable(torch.from_numpy(np.full((num_y, 1), 1 / num_y))).float().to(device)

        re_sp, _, distri_0, distri_k, latents_k, output_sh_loss, log_var, _ = MLA_model(batch_day_0, batch_day_k, p, q,
                                                                                        train_flag=False)

        ot_loss = ot_weight * output_sh_loss

        latents_k = latents_k[:, None, :, :]
        latents_k = torch.transpose(latents_k, 3, 2)

        batch_size = latents_k.shape[0]
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(latents_k, device=device)

        z_noisy = q_sample(x_start=latents_k, t=t, noise=noise)
        predicted_noise = diff_model(z_noisy, t)
        diffusion_loss = appro_alpha * F.mse_loss(noise, predicted_noise)

        total_loss = ot_loss + diffusion_loss

        total_loss.backward(retain_graph=True)

        optimizer.step()

    with torch.no_grad():
        num_x, num_y = spike_day_0.shape[0], spike_day_k.shape[0]

        p = Variable(torch.from_numpy(np.full((num_x, 1), 1 / num_x))).float().to(device)
        q = Variable(torch.from_numpy(np.full((num_y, 1), 1 / num_y))).float().to(device)

        if epoch % 5 == 0 or epoch == epochs - 1:
            current_metric = float(
                logger_performance(MLA_model, spike_day_0, spike_day_k, p, q_test, test_trial_vel_tide))
            if current_metric > best_metric:
                best_metric = current_metric

            # if total_loss < pre_total_loss_:
            torch.save(MLA_model.state_dict(), '../model_checkpoints/vae_model_mla.pth')
            pre_total_loss_ = total_loss

            # Testing Phase
            _, _, _, _, test_latents, _, _, x_after_lowd = MLA_model(spike_day_0, spike_day_k, p, q_test,
                                                                     train_flag=False)
            test_latents = np.array(test_latents.cpu())

            vanilla_model_dict = torch.load('../model_checkpoints/vae_model_mla.pth', weights_only=True,
                                            map_location=torch.device('cpu'))

            VAE_Readout_model = VAE_Readout_Model(len_trial, num_neurons_s, num_neurons_t, vel_dim)
            DL_dict_keys = VAE_Readout_model.state_dict().keys()
            vanilla_model_dict_keys = vanilla_model_dict.keys()

            DL_dict_new = VAE_Readout_model.state_dict().copy()

            for key in vanilla_model_dict_keys:
                DL_dict_new[key] = vanilla_model_dict[key]

            VAE_Readout_model.load_state_dict(DL_dict_new)

            r2, rmse = vel_cal(test_trial_vel_tide, VAE_Readout_model, torch.Tensor(test_latents), x_after_lowd)
            print(f"Epoch: {epoch:4d} {' ' * 10} loss: {total_loss.item():0.4f} {' ' * 10} "
                  f"R2: {r2:0.4f} {' ' * 10} RMSE: {rmse:0.4f}")

            if epoch % 100 == 0 or epoch == epochs - 1:
                # best_metric
                print(f"Best_Metric at {epoch} is : {best_metric:0.4f}")
