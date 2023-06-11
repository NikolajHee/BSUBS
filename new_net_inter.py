import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#from torchsummary import summary
from tqdm import tqdm
import os
from new_net import log_Normal, log_standard_Normal, encoder, decoder, VAE, generate_image, plot_reconstruction, plot_1_reconstruction
from dataloader import BBBC

def interpolate(v1, v2, Nstep):
    for i in range(Nstep):
        r = v2 - v1
        v = v1 + r * (i / (Nstep - 1))
        yield v

class get_image_based_on_id(BBBC):
    def __init__(self, folder_path, meta_path, test=False, normalize='to_1', exclude_dmso=False, shuffle=False):
        subset = (1,1)
        super().__init__(folder_path, meta_path, subset, test, normalize, exclude_dmso, shuffle)
        try:
            self.meta = pd.read_csv(meta_path, index_col=0)
        except:
            raise ValueError("Please change variable 'main_path' to the path of the data folder (should contain metadata.csv, ...)")
    
    def __getitem__(self, idx): 
        img_name = os.path.join(self.folder_path,
                                self.meta[self.col_names[1]].iloc[idx], 
                                self.meta[self.col_names[3]].iloc[idx])
    
        image = np.load(img_name)

        # convert the data to appropriate format
        if self.normalize == 'to_1':
            image = self.normalize_to_1(image)
        else:
            image = self.normalize_to_255(image)

        moa = self.meta[self.col_names[-1]].iloc[idx]
        compound = self.meta[self.col_names[-3]].iloc[idx]
        id = self.meta.index[idx]

        sample = {"id": id, 
                  "image": torch.tensor(image), 
                  "moa": moa, 
                  "compound": compound,
                  }

        return sample
    


index_image_1 = 452305
index_image_2 = 475106


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 200
epochs = 200
batch_size = 200

input_dim = 68
channels = 3

train_size = 20000
test_size = 1000

#epochs, batch_size, train_size = 2, 1, 10

# torch.backends.cudnn.deterministic = True
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

from dataloader import BBBC

main_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"
#main_path = "/Users/nikolaj/Fagprojekt/Data/"


exclude_dmso = False
shuffle = True

subset = (train_size, test_size)

dataset_train = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                        meta_path=main_path + "metadata.csv",
                        subset=subset,
                        test=False,
                        exclude_dmso=exclude_dmso,
                        shuffle=shuffle)

dataset_test = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                        meta_path=main_path + "metadata.csv",
                        subset=subset,
                        test=True,
                        exclude_dmso=exclude_dmso,
                        shuffle=shuffle)


X_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
X_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

VAE = VAE(
    latent_dim=latent_dim,
    input_dim=input_dim,
    channels=channels,
    hidden_channels=[8,16,32]
).to(device)

#print("VAE:")
#summary(VAE, input_size=(channels, input_dim, input_dim))

encoder_VAE, decoder_VAE, REs, KLs, ELBOs = VAE.train_VAE(
    dataloader=X_train, epochs=epochs)


results_folder = 'interpolation/'
if not(os.path.exists(results_folder)):
    os.mkdir(results_folder)



x = np.arange(0, len(REs), 1)

plt.plot(x, REs, label="Reconstruction Error")
plt.plot(x, KLs, label="Regularizer")
plt.plot(x, ELBOs, label="ELBO")
plt.xlabel("iterationns")
plt.title("ELBO Components")
plt.legend()
plt.savefig(results_folder + 'ELBO_components.png')
plt.show()


get_image = get_image_based_on_id(main_path + "singh_cp_pipeline_singlecell_images", main_path + "metadata.csv")

_, _, _, z1 = VAE(torch.tensor(get_image[index_image_1]["image"]).unsqueeze(0).to(device), save_latent=True)
_, _, _, z2 = VAE(torch.tensor(get_image[index_image_2]["image"]).unsqueeze(0).to(device), save_latent=True)


generated_images = []

for z in interpolate(z1, z2, 9):
    mean = VAE.decode(z)
    image = torch.normal(mean=mean, std=0.05).to(device)
    image = image.view(channels, input_dim, input_dim)
    image = image.clip(0,1).detach().cpu().numpy()  
    generated_images.append(image)

fig, ax = plt.subplots(1, 11, figsize=(20, 2))
ax[0].imshow(get_image[index_image_1]["image"].numpy().reshape((68,68,3)))
ax[0].axis('off')
for i in range(9):
    ax[i+1].imshow(generated_images[i].transpose(1,2,0))
    ax[i+1].axis('off')
ax[10].imshow(get_image[index_image_2]["image"].numpy().reshape((68,68,3)))
ax[10].axis('off')
plt.tight_layout()
plt.title("Interpolation between two images")
plt.savefig(results_folder + 'interpol.png')
plt.show()

for i, image in enumerate(X_train.dataset):
    if i == 10:
        break
    plot_1_reconstruction(image['image'],
                        vae = VAE,
                        name="training" + '\n' + str(image['id']), 
                        results_folder=results_folder, 
                        latent_dim=latent_dim, 
                        channels=channels, 
                        input_dim=input_dim)
    

for i, image in enumerate(X_test.dataset):
    if i == 10:
        break
    plot_1_reconstruction(image['image'],
                        vae = VAE,
                        name="test" + '\n' + str(image['id']), 
                        results_folder=results_folder, 
                        latent_dim=latent_dim, 
                        channels=channels, 
                        input_dim=input_dim)
