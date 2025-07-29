from notebook_utils.data_prepare import load_dummy_dataset
from model_functions.vae import VAE_Model

# day_data, day_label = load_tensors_by_index(i=0, dataset_folder='')
day_data, day_label = load_dummy_dataset(200)  # day_data.shape: torch.Size([200, 95]), day_label.shape: torch.Size([200, 6])

print(day_data.shape, day_label.shape)

