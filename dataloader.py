# Importing the libraries
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os



### Following dataloader is made by the following guide on 
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class BBBC(Dataset):
    def __init__(self, folder_path, meta_path, 
                 subset=(390716, 97679), # 1/5 of the data is test data by default
                test=False, normalize='to_1',
                exclude_dmso=False):
        try:
            self.meta = pd.read_csv(meta_path, index_col=0)
        except:
            raise ValueError("Please change variable 'main_path' to the path of the data folder (should contain metadata.csv, ...)")
        self.col_names = self.meta.columns
        self.folder_path = folder_path
        self.train_size, self.test_size = subset
        self.test = test  
        self.normalize = normalize  

        if exclude_dmso: self.exclude_dmso()
        self.meta = self.meta.iloc[self.train_size:self.train_size + self.test_size, :] if self.test else self.meta.iloc[:self.train_size,:]
        
    def __len__(self,):
        return self.test_size if self.test else self.train_size
    
    def normalize_to_255(self, x):
        # helper function to normalize to {0,1,...,255}
        to_255 = (x/np.max(x)) * 255
        return to_255.astype(np.uint8).reshape((3,68,68))

    def normalize_to_1(self, x):
        # helper function to normalize to [0...1]
        to_1 = (x/np.max(x))
        return to_1.astype(np.float32).reshape((3,68,68))   
    
    def exclude_dmso(self):
        # helper function to exclude DMSO from the dataset
        self.meta = self.meta[self.meta[self.col_names[-1]] != 'DMSO']

    def __getitem__(self, idx):
        #if self.test: idx += self.train_size
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

        sample = {"idx": idx, 
                  "image": torch.tensor(image), 
                  "moa": moa, 
                  "compound": compound,
                  }

        return sample


if __name__ == "__main__":
    batch_size = 5

    train_size = 25
    test_size = 25

    exclude_dmso = True

    # path to singlecells

    main_path = "/Users/nikolaj/Fagprojekt/Data/"
    
    # on hpc:
    # main_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"

    subset = (train_size, test_size)

    dataset_train = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                            meta_path=main_path + "metadata.csv",
                            subset=subset,
                            test=False,
                            exclude_dmso=exclude_dmso)

    dataset_test = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                            meta_path=main_path + "metadata.csv",
                            subset=subset,
                            test=True,
                            exclude_dmso=exclude_dmso)

    X_train = DataLoader(dataset_train, batch_size=batch_size)#, shuffle=True, num_workers=0)

    X_test = DataLoader(dataset_test, batch_size=batch_size)#, shuffle=True, num_workers=0)  



    # plot number of batches of size batch_size in one plot

    num_batches = 5

    fig, axs = plt.subplots(num_batches, batch_size, figsize=(10, 8))

    for i, batch in enumerate(X_train):
        if i == num_batches:
            break
        for j, sample in enumerate(batch["image"]):
            print("batch: {}, image: {}, moa: {}, compound: {}".format(i,batch['idx'][j], batch['moa'][j], batch['compound'][j]))
            axs[i,j].imshow(sample.reshape(68,68,3), cmap="gray")
            axs[i,j].set_title("batch: {}, image: {}".format(i,batch['idx'][j]))
            axs[i,j].axis("off")
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(num_batches, batch_size, figsize=(10, 8))

    for i, batch in enumerate(X_test):
        if i == num_batches:
            break
        for j, sample in enumerate(batch["image"]):
            print("batch: {}, image: {}, moa: {}, compound: {}".format(i,batch['idx'][j], batch['moa'][j], batch['compound'][j]))
            axs[i,j].imshow(sample.reshape(68,68,3), cmap="gray")
            axs[i,j].set_title("batch: {}, image: {}".format(i,batch['idx'][j]))
            axs[i,j].axis("off")
    plt.tight_layout()
    plt.show()