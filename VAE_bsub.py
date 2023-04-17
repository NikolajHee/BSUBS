# imports
from VAE import log_Categorical, log_Normal, log_standard_Normal, encoder, decoder, VAE, generate_image
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import torchvision.transforms as T

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# configuration
latent_dim = 50
epochs = 5
batch_size = 10

pixel_range = 256
input_dim = 28
channels = 3

train_size = 10000
test_size = 1000


# dataset class
class DataLoader(Dataset):
    def __init__(self, folder_path, metadata_path, subset, test = False, transform=None):
        """
        Parameters
        ----------
        folder_path : str
            Path to folder containing images.
        metadata_path : str
            Path to metadata file.
        subset : list, optional
            Tuple of indices for training and testing (train_size, test_size)
        test : bool, optional
            Whether to use the test set. The default is False.
        """

        self.meta_data = pd.read_csv(metadata_path, index_col=0)
        self.col_names = self.meta_data.columns
        self.folder_path = folder_path
        self.train_size, self.test_size = subset
        self.test = test
        self.transform = transform

        if test:
            self.meta_data = self.meta_data.iloc[self.train_size:self.train_size + self.test_size, :]
        else:
            self.meta_data = self.meta_data.iloc[:self.train_size,:]
        
    
    def __len__(self,):
        return len(self.meta_data)
    
    def normalize_to_255(self, x):
        normalize = (x-np.min(x))/(np.max(x)-np.min(x))
        to_255 = torch.tensor(np.round(normalize*255)).view((1,3,68,68)).float()
        if self.transform: to_255 = self.transform(to_255)
        return to_255

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        if type(idx) == int:
            i = idx + self.train_size if self.test else idx
            img_name = os.path.join(self.folder_path,
                                    self.meta_data[self.col_names[1]][i], 
                                    self.meta_data[self.col_names[3]][i])
            image = np.load(img_name)
            return self.normalize_to_255(image)
        
        start = idx.start + self.train_size if self.test else idx.start
        stop = idx.stop + self.train_size if self.test else idx.stop
        for i in range(start, stop):
            img_name = os.path.join(self.folder_path,
                                    self.meta_data[self.col_names[1]][i], 
                                    self.meta_data[self.col_names[3]][i])
            image = np.load(img_name)
            if i == start:
                sample = self.normalize_to_255(image)
            else:
                sample = torch.cat((sample, self.normalize_to_255(image)), dim=0)
        return sample

# (batch_size, channels, input_dim, input_dim)

subset = (train_size,test_size)




main_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"

folder_path = os.path.join(main_path, "singh_cp_pipeline_singlecell_images")
metadata_path= os.path.join(main_path, "metadata.csv")
output_path = 'resulting_images.npy'



X_train = DataLoader(folder_path = folder_path,
                     metadata_path=metadata_path,
                     subset=subset, transform=T.Resize(size=(28,28)))  

X_test = DataLoader(folder_path = folder_path,
                     metadata_path=metadata_path,
                     subset=subset, test=True, transform=T.Resize(size=(28,28)))  

print("sucessfully initialized dataloader")

VAE = VAE(X_train, pixel_range=pixel_range,
        latent_dim=latent_dim, input_dim=input_dim, channels=channels).to(device)

print("sucessfully initialized VAE")

encoder_VAE, decoder_VAE, reconstruction_errors, regularizers, latent_space = VAE.train_VAE(
        X=X_train, epochs=epochs, batch_size=batch_size)

print("sucessfully trained VAE")


np.savez("latent_space.npz", latent_space=latent_space.detach().numpy())

generated_images = []
for i in range(9):
    image = generate_image(X_test[i], encoder_VAE, decoder=decoder_VAE,
                        latent_dim=latent_dim, channels=channels, input_dim=input_dim)
    generated_images.append(image)

generated_images = np.array(generated_images)

np.save("generated_images", generated_images)


# saving
torch.save(encoder_VAE, "encoder.pt")
torch.save(decoder_VAE, "decoder.pt")




