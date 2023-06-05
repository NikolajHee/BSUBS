# imports
from VAE2 import log_Categorical, log_Normal, log_standard_Normal, encoder, decoder, VAE, generate_image
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import pickle
# from torchsummary import summary
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# TODO: Add multiple initializations of VAE and save the best one

print("imports done")

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# configuration
latent_dim = 16
epochs = 1
batch_size = 10

pixel_range = 256//2
input_dim = 68
channels = 3

train_size = 100
test_size = 10

save_folder = 'BBC_VAE_results/'


if not(os.path.exists(save_folder)):
    os.mkdir(save_folder)


subset = (train_size, test_size)

# dataset class



# Custom dataset class
class BBBC(Dataset):
    def __init__(self, folder_path, meta_path, subset=(390716, 97679), # 1/5 of the data is test data  by default
                                                          test=False):
        self.meta = pd.read_csv(meta_path, index_col=0)
        self.col_names = self.meta.columns
        self.folder_path = folder_path
        self.train_size, self.test_size = subset
        self.test = test    
        self.meta = self.meta.iloc[self.train_size:self.train_size + self.test_size, :] if self.test else self.meta.iloc[:self.train_size,:]
        
    def __len__(self,):
        return self.test_size if self.test else self.train_size
    
    def normalize_to_255(self, x):
        # helper function to normalize to 255
        to_255 = (x/np.max(x)) * 255
        return to_255.astype(np.uint8).reshape((3,68,68))

    def __getitem__(self, idx):
        if self.test: idx += self.train_size
        img_name = os.path.join(self.folder_path,
                                self.meta[self.col_names[1]][idx], 
                                self.meta[self.col_names[3]][idx])
        image = np.load(img_name)
        image = self.normalize_to_255(image)
        moa = self.meta[self.col_names[-1]][idx]
        # more relevant stuff here

        sample = {"idx": idx, 
                  "image": image, 
                  "moa": moa, 
                  } # more relevant stuff here

        return sample

# (batch_size, channels, input_dim, input_dim)


main_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"
# main_path = "/Users/nikolaj/Fagprojekt/Data/"

folder_path = os.path.join(main_path, "singh_cp_pipeline_singlecell_images")
meta_path= os.path.join(main_path, "metadata.csv")


print("script started")




dataset_train = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                        meta_path=main_path + "metadata.csv",
                        subset=subset,
                        test=False)

dataset_test = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                        meta_path=main_path + "metadata.csv",
                        subset=subset,
                        test=True)

X_train = DataLoader(dataset_train, batch_size=batch_size)#, shuffle=True, num_workers=0)

X_test = DataLoader(dataset_test, batch_size=batch_size)#, shuffle=True, num_workers=0)  


print("sucessfully initialized dataloader")

VAE_ = VAE(X_train, pixel_range=pixel_range,
        latent_dim=latent_dim, input_dim=input_dim, channels=channels).to(device)

# print("VAE:")
# summary(VAE, input_size=(channels, input_dim, input_dim))

print("sucessfully initialized VAE")

encoder_VAE, decoder_VAE, reconstruction_errors, regularizers, latent_space, error_log = VAE_.train_VAE(
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
