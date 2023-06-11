
import numpy as np
import torch
from dataloader import get_image_based_on_id
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def interpolate_between_two_images(vae, index_image_1, index_image_2, main_path, results_folder=''):

    def interpolate(v1, v2, Nstep):
        for i in range(Nstep):
            r = v2 - v1
            v = v1 + r * (i / (Nstep - 1))
            yield v

    channels = vae.channels
    input_dim = vae.input_dim

    get_image = get_image_based_on_id(main_path + "singh_cp_pipeline_singlecell_images", main_path + "metadata.csv")
    _, _, _, z1 = vae(get_image[index_image_1]["image"].clone().detach().float().view(1,3,68,68).to(device), save_latent=True)
    _, _, _, z2 = vae(get_image[index_image_2]["image"].clone().detach().float().view(1,3,68,68).to(device), save_latent=True)


    generated_images = []

    for z in interpolate(z1, z2, 9):
        mean = vae.decode(z)
        image = torch.normal(mean=mean, std=0.05).to(device)
        image = image.view(channels, input_dim, input_dim)
        image = image.clip(0,1).detach().cpu().numpy()  
        generated_images.append(image)

    fig, ax = plt.subplots(1, 11, figsize=(10, 3))
    ax[0].imshow(get_image[index_image_1]["image"].reshape((68,68,3)))
    ax[0].axis('off')
    ax[0].set_title("Original\nimage 1")
    for i in range(9):
        ax[i+1].imshow(generated_images[i].reshape((68,68,3)))
        ax[i+1].axis('off')
    ax[10].imshow(get_image[index_image_2]["image"].reshape((68,68,3)))
    ax[10].axis('off')
    ax[10].set_title("Original\nimage 2")

    fig.suptitle("Interpolation between two images", fontsize=16)
    plt.tight_layout()
    #plt.title("Interpolation between two images")
    plt.savefig(results_folder + 'interpol.png')
    plt.show()



def interpolate_between_three_images(vae, index_image_1, index_image_2, index_image_3, main_path, results_folder=''):
    
    channels = vae.channels
    input_dim = vae.input_dim

    get_image = get_image_based_on_id(main_path + "singh_cp_pipeline_singlecell_images", main_path + "metadata.csv")
    _, _, _, z1 = vae(get_image[index_image_1]["image"].clone().detach().float().view(1,3,68,68).to(device), save_latent=True)
    _, _, _, z2 = vae(get_image[index_image_2]["image"].clone().detach().float().view(1,3,68,68).to(device), save_latent=True)
    _, _, _, z3 = vae(get_image[index_image_3]["image"].clone().detach().float().view(1,3,68,68).to(device), save_latent=True)

    def interpolate(v1, v2, Nstep):
        for i in range(Nstep):
            r = v2 - v1
            v = v1 + r * (i / (Nstep - 1))
            yield v

    generated_images = [[] for _ in range(9)]
    
    for i, z_ in enumerate(interpolate(z1, z2, 9)):
        for z in interpolate(z1+z_, z3+z_, 9):
            mean = vae.decode(z)
            image = torch.normal(mean=mean, std=0.05).to(device)
            image = image.view(channels, input_dim, input_dim)
            image = image.clip(0,1).detach().cpu().numpy()  
            generated_images[i].append(image)
    
    fig, ax = plt.subplots(9, 9, figsize=(10, 10))
    for i in range(9):  
        for j in range(9):
            ax[i,j].imshow(generated_images[i][j].reshape((68,68,3)))
            ax[i,j].axis('off')
    fig.suptitle("Interpolation between three images", fontsize=16)
    plt.tight_layout()
    plt.savefig(results_folder + 'interpol_grid.png')
    plt.show()




