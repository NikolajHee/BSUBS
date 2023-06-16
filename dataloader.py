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
                exclude_dmso=False,
                shuffle=False):
        try:
            self.meta = pd.read_csv(meta_path, index_col=0)
        except:
            # this could also be other errors, but this is the most likely
            raise ValueError("Please change variable 'main_path' to the path of the data folder (should contain metadata.csv, ...)")
        

        self.col_names = self.meta.columns
        self.folder_path = folder_path
        self.train_size, self.test_size = subset
        self.test = test  
        self.normalize = normalize  

        if shuffle: self.meta = self.meta.sample(frac=1)
        if exclude_dmso: self.exclude_dmso()
        self.meta = self.meta.iloc[self.train_size:self.train_size + self.test_size, :] if self.test else self.meta.iloc[:self.train_size,:]
        
        
    def __len__(self,):
        return self.test_size if self.test else self.train_size
    

    def label_encoder(self, x):
        classes = np.array(['DMSO', 'Actin disruptors', 'Aurora kinase inhibitors',
       'Cholesterol-lowering', 'DNA damage', 'DNA replication',
       'Eg5 inhibitors', 'Epithelial', 'Kinase inhibitors',
       'Microtubule destabilizers', 'Microtubule stabilizers',
       'Protein degradation', 'Protein synthesis'])
        
        return np.where(classes == x)[0][0]
    
    def label_decoder(self, x):
        classes = np.array(['DMSO', 'Actin disruptors', 'Aurora kinase inhibitors',
       'Cholesterol-lowering', 'DNA damage', 'DNA replication',
       'Eg5 inhibitors', 'Epithelial', 'Kinase inhibitors',
       'Microtubule destabilizers', 'Microtubule stabilizers',
       'Protein degradation', 'Protein synthesis'])
        
        return classes[x]


    def normalize_to_1(self, x):
        # helper function to normalize to [0...1]
        to_1 = ((x-np.min(x,axis=(0,1)))/np.max(x,axis=(0,1)))
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
        id = self.meta.index[idx]

        sample = {"id": id, 
                  "image": torch.tensor(image), 
                  "moa": self.label_encoder(moa), 
                  "compound": compound,
                  "moa_name": moa,
                  }

        return sample


class get_image_based_on_id(BBBC):
    # class to get a picture based on the id solely
    # (used when doing interpolation, as we want specific images)
    def __init__(self, folder_path, meta_path, normalize='to_1'):
        subset = (1,1)
        exclude_dmso=False
        shuffle=False
        test=False


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
                  "moa": self.label_encoder(moa), 
                  "compound": compound,
                  "moa_name": moa,
                  }

        return sample
    

if __name__ == "__main__":
    batch_size = 5

    train_size = 100_000
    test_size = 10_000

    exclude_dmso = False
    shuffle=True

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    # path to singlecells

    main_path = "/Users/nikolaj/Fagprojekt/Data/"
    
    # on hpc:
    # main_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"

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

    X_train = DataLoader(dataset_train, batch_size=batch_size)#, shuffle=True, num_workers=0)

    X_test = DataLoader(dataset_test, batch_size=batch_size)#, shuffle=True, num_workers=0)  

    
    # distribution of classes in train and test set
    print("Train set:")
    print(dataset_train.meta[dataset_train.col_names[-1]].value_counts())
    print("Test set:")
    print(dataset_test.meta[dataset_test.col_names[-1]].value_counts())

    # 157116
    # 381204
    # 366536
    # BAD 117772
    # NOISE 133205
    # NOISE 134557
    # multiple cells 428546
    # multiple cells 35983

    # save the cells
    bad_cells = [378294, 424125, 207542, 468092, 468623, 10158, 414951]

    get_image = get_image_based_on_id(main_path + "singh_cp_pipeline_singlecell_images",
                                      main_path + "metadata.csv",
                                      normalize='to_1')

    for cell in bad_cells:
        sample = get_image[cell]
        plt.imshow(sample['image'].reshape((68,68,3)))
        plt.title(sample['moa'] + " - " + str(sample['id']))
        plt.savefig("plots/bad_cells/" + str(cell) + ".png")
        plt.show()

    for i in range(1000):
        sample = dataset_test[i+1300]
        plt.imshow(sample['image'].reshape((68,68,3)))
        plt.title(sample['moa_name'] + " - " + str(sample['id']))
        plt.show()


# perfect round : 317244 (microtubule stabilizers)


# interpolation
# microtubule destabilizers 200276
# microtubule destabilizers 136275

# interpolation
# microtubule stabiliezrs 461163
# microtubule stabiliezrs 166720


# interpolation
# microtubule stabilizers 308072
# microtubule stabilizers 328098

# interpolation
# dmso 322827
# dmso 219669
# dmso 473201

# interpolation
# dmso 219669
# dmso 358837

# interpolation
# DNA replication 115811

# interpolation
# epithelial 23742


# interpolation
# aurora kinase inhibitors 71927
# aurora kinase inhibitors 246443

# BAD CELL PICTURES
# 157116
# 381204
# 366536
# BAD 117772
# NOISE 133205
# NOISE 134557
# multiple cells 428546
# multiple cells 35983
# 371043
