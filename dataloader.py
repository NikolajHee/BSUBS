from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import torch


class BBBC(Dataset):
    def __init__(self, folder_path, meta_path, subset=(390716, 97679), # 1/5 of the data is test data  by default
                                                          test=False, normalize='to_1'):
        self.meta = pd.read_csv(meta_path, index_col=0)
        self.col_names = self.meta.columns
        self.folder_path = folder_path
        self.train_size, self.test_size = subset
        self.test = test  
        self.normalize = normalize  
        self.meta = self.meta.iloc[self.train_size:self.train_size + self.test_size, :] if self.test else self.meta.iloc[:self.train_size,:]
        
    def __len__(self,):
        return self.test_size if self.test else self.train_size
    
    def normalize_to_255(self, x):
        # helper function to normalize to 255
        to_255 = (x/np.max(x)) * 255
        return to_255.astype(np.uint8).reshape((3,68,68))

    def normalize_to_1(self, x):
        # helper function to normalize to 255
        to_1 = (x/np.max(x))
        return to_1.astype(np.float32).reshape((3,68,68))   

    def __getitem__(self, idx):
        if self.test: idx += self.train_size
        img_name = os.path.join(self.folder_path,
                                self.meta[self.col_names[1]][idx], 
                                self.meta[self.col_names[3]][idx])
        image = np.load(img_name)
        if self.normalize == 'to_1':
            image = self.normalize_to_1(image)
        else:
            image = self.normalize_to_255(image)


        moa = self.meta[self.col_names[-1]][idx]
        # more relevant stuff here

        sample = {"idx": idx, 
                  "image": torch.tensor(image), 
                  "moa": moa, 
                  } # more relevant stuff here

        return sample