# imports
from VAE import log_Categorical, log_Normal, log_standard_Normal, encoder, decoder, VAE, generate_image
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import pickle

# TODO: Add multiple initializations of VAE and save the best one

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# configuration
latent_dim = 10
epochs = 5
batch_size = 10

pixel_range = 256
input_dim = 28
channels = 3

train_size = 10000
test_size = 1000

save_folder = 'BBC_VAE_results/'

subset = (train_size,test_size)

# dataset class
class DataLoader(Dataset):
    def __init__(self, folder_path, meta_path, subset, test=False):
        self.meta = pd.read_csv(meta_path, index_col=0)
        self.col_names = self.meta.columns
        self.folder_path = folder_path
        self.train_size, self.test_size = subset
        self.test = test    
        self.meta = self.meta.iloc[self.train_size:self.train_size + self.test_size, :] if self.test else self.meta.iloc[:self.train_size,:]
        
    def __len__(self,):
        return self.test_size if self.test else self.train_size
    
    def normalize_to_255(self, x):
        normalize = (x-np.min(x))/(np.max(x)-np.min(x))
        to_255 = torch.tensor(np.round(normalize*255)).view((1,3,68,68)).float()
        return to_255

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        if type(idx) == int: idx = slice(idx, idx+1)
        
        start = idx.start + self.train_size if self.test else idx.start
        stop = idx.stop + self.train_size if self.test else idx.stop
        for i in range(start, stop):
            img_name = os.path.join(self.folder_path,
                                    self.meta[self.col_names[1]][i], 
                                    self.meta[self.col_names[3]][i])
            image = np.load(img_name)
            if i == start:
                sample = self.normalize_to_255(image)
            else:
                sample = torch.cat((sample, self.normalize_to_255(image)), dim=0)
        return sample

# (batch_size, channels, input_dim, input_dim)


main_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"

folder_path = os.path.join(main_path, "singh_cp_pipeline_singlecell_images")
metadata_path= os.path.join(main_path, "metadata.csv")




X_train = DataLoader(folder_path = folder_path,
                     metadata_path=metadata_path,
                     subset=subset)  

X_test = DataLoader(folder_path = folder_path,
                     metadata_path=metadata_path,
                     subset=subset, test=True)  

print("sucessfully initialized dataloader")

VAE = VAE(X_train, pixel_range=pixel_range,
        latent_dim=latent_dim, input_dim=input_dim, channels=channels).to(device)

print("sucessfully initialized VAE")

encoder_VAE, decoder_VAE, reconstruction_errors, regularizers, latent_space, error_log = VAE.train_VAE(
        X=X_train, epochs=epochs, batch_size=batch_size)

print("sucessfully trained VAE")


np.savez(save_folder +  "latent_space.npz", latent_space=latent_space.detach().numpy())

np.savez(save_folder + "reconstruction_errors.npz", reconstruction_errors=reconstruction_errors.detach().numpy())

np.savez(save_folder + "regularizers.npz", regularizers=regularizers.detach().numpy())

generated_images = []
for i in range(9):
    image = generate_image(X_test[i], encoder_VAE, decoder=decoder_VAE,
                        latent_dim=latent_dim, channels=channels, input_dim=input_dim)
    generated_images.append(image)

generated_images = np.array(generated_images)

np.save(save_folder + "generated_images", generated_images)



# save error_log
with open(save_folder + "error_log.pkl", "wb") as f:
    pickle.dump(error_log, f)
