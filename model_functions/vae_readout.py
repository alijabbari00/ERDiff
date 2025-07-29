import torch
from torch import nn

# Own Dataset
len_trial, num_neurons_s, num_neurons_t = 14, 187, 172


# Public Dataset
# len_trial, num_neurons_s, num_neurons_t = 25, 95, 95

class VAE_Readout_Model(nn.Module):
    def __init__(self, spike_dim_s: int = num_neurons_s,
                 spike_dim_t: int = num_neurons_t,
                 vel_dim: int = 2):
        """Decoder used after alignment.

        Parameters
        ----------
        spike_dim_s : int
            Number of source neurons.
        spike_dim_t : int
            Number of target neurons.
        vel_dim : int
            Dimensionality of the velocity output.
        """

        super(VAE_Readout_Model, self).__init__()
        # Hyper-Parameters
        self.spike_dim_s = spike_dim_s
        self.spike_dim_t = spike_dim_t
        self.low_dim = 64
        self.latent_dim = 8
        self.vel_dim = vel_dim
        self.encoder_n_layers, self.decoder_n_layers = 1, 1
        self.hidden_dims = [64, 32]
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()

        # Alignment Structure
        self.low_d_readin_s = nn.Linear(self.spike_dim_s, self.low_dim)

        self.align_layer = nn.Linear(self.spike_dim_t, self.spike_dim_t, bias=False)

        self.low_d_readin_t_1 = nn.Linear(self.spike_dim_t, self.spike_dim_t)

        self.low_d_readin_t_2 = nn.Linear(self.spike_dim_t, self.low_dim)

        # self.al_fc2 = nn.Linear(self.spike_dim, self.spike_dim)

        nn.init.eye_(self.align_layer.weight)
        # nn.init.eye_(self.al_fc2.weight)

        # Encoder Structure
        self.encoder_rnn = nn.RNN(self.low_dim, self.hidden_dims[0], self.encoder_n_layers,
                                  bidirectional=False, nonlinearity='tanh', batch_first=True)
        for name, param in self.encoder_rnn.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, 0.1)

        # self.encoder_rnn_bn = nn.BatchNorm1d(len_trial)

        self.fc_mu_1 = nn.Linear(self.hidden_dims[0], self.latent_dim)
        # self.fc_mu_2 = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        self.fc_log_var_1 = nn.Linear(self.hidden_dims[0], self.latent_dim)
        # self.fc_log_var_2 = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        # Spike Decoder Structure
        self.sde_rnn = nn.RNN(self.latent_dim, self.latent_dim, self.decoder_n_layers, bidirectional=False,
                              nonlinearity='tanh', batch_first=True)
        self.sde_fc1 = nn.Linear(self.latent_dim, self.hidden_dims[0])
        self.sde_fc2 = nn.Linear(self.hidden_dims[0], self.spike_dim_s)

        # Velocity Decoder Structure
        self.vde_rnn = nn.RNN(self.latent_dim, self.latent_dim, self.decoder_n_layers, bidirectional=False,
                              nonlinearity='tanh', batch_first=True)
        for name, param in self.vde_rnn.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, 0.1)

        self.vde_fc_minus_0 = nn.Linear(self.latent_dim, self.vel_dim, bias=False)
        self.vde_fc_minus_1 = nn.Linear(self.latent_dim, self.vel_dim, bias=False)
        self.vde_fc_minus_2 = nn.Linear(self.latent_dim, self.vel_dim, bias=False)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, train_flag):
        # Spike Decoder
        re_sp, _ = self.sde_rnn(x)
        re_sp = self.sde_fc1(re_sp)
        re_sp = self.softplus(self.sde_fc2(re_sp))

        # Velocity Decoder
        vel_latent = x
        vel_hat_minus_0 = self.vde_fc_minus_0(vel_latent)
        vel_hat_minus_1 = self.vde_fc_minus_1(vel_latent)
        vel_hat_minus_2 = self.vde_fc_minus_2(vel_latent)

        vel_hat = torch.zeros_like(vel_hat_minus_0)

        for i in range(len_trial):
            vel_hat[:, i, :] += vel_hat_minus_0[:, i, :]

        return re_sp, vel_hat
