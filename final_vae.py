import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#from torchsummary import summary
from tqdm import tqdm
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_Normal(x, mu, log_var):
    D = x.shape[1]
    log_p = -0.5 * ((x - mu) ** 2.0 * torch.exp(-log_var) +
                    log_var + D * torch.log(2 * torch.tensor(np.pi)))
    return log_p


def log_standard_Normal(x):
    D = x.shape[1]
    log_p = -0.5 * (x**2.0 + D * torch.log(2 * torch.tensor(np.pi)))
    return log_p


class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels):
        super(encoder, self).__init__()
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(
            16 * self.input_dim * self.input_dim, 2 * latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.01)(x)
        x = x.view(-1, 16 * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        return x


class decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels):
        super(decoder, self).__init__()
        self.input_dim = input_dim
        self.channels = channels

        self.input = nn.Linear(
            latent_dim,   16 * self.input_dim * self.input_dim)
        self.conv2 = nn.ConvTranspose2d(
            16, channels, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.output = nn.Linear(channels * self.input_dim * self.input_dim,
                                2 * channels * self.input_dim * self.input_dim)


    def forward(self, x):
        x = self.input(x)
        x = nn.LeakyReLU(0.01)(x)
        x = x.view(-1, 16, self.input_dim, self.input_dim)
        x = self.conv2(x)
        x = nn.LeakyReLU(0.01)(x)
        x = x.view(-1,  self.channels * self.input_dim * self.input_dim)
        x = self.output(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim, input_dim, channels, beta = 1.0):
        super(VAE, self).__init__()
        self.encoder = encoder(input_dim, latent_dim, channels)
        self.decoder = decoder(input_dim, latent_dim, channels)
        self.latent_dim = latent_dim
        self.channels = channels
        self.input_dim = input_dim
        self.beta = beta
        self.var = 0

    def encode(self, x):
        mu, log_var = torch.split(
            self.encoder.forward(x), self.latent_dim, dim=1)
        return mu, log_var

    def reparameterization(self, mu, log_var):
        self.eps = torch.normal(mean=0, std=torch.ones(self.latent_dim)).to(device)
        return mu + torch.exp(0.5 * log_var) * self.eps

    def decode(self, z):
        mu, log_var = torch.split(
            self.decoder.forward(z), self.channels * self.input_dim * self.input_dim, dim=1)
        std = torch.exp(log_var)
        return mu, std

    def forward(self, x, save_latent = False):
        #* kl divergence
        mu, log_var = self.encode(x)
        z = self.reparameterization(mu, log_var)


        log_posterior = log_Normal(z, mu, log_var)

        log_prior = log_standard_Normal(z)
        
        regularizer = - torch.sum(log_prior - log_posterior, dim=-1).mean() 

        #* reconstruction error
        decode_mu, decode_var = self.decode(z)

        # save variance for printing
        self.var = decode_var

        log_like = (1 / (2 * (decode_var)) * nn.functional.mse_loss(decode_mu, x.flatten(
            start_dim=1, end_dim=-1), reduction="none")) + 0.5 * torch.log(decode_var) + 0.5 * torch.log(2 * torch.tensor(np.pi))

        reconstruction_error = torch.sum(log_like, dim=-1).mean()
        

        elbo = reconstruction_error + regularizer * self.beta

        #tqdm.write(
        #    f"ELBO: {elbo.item()}, Reconstruction error: {reconstruction_error.item()}, Regularizer: {regularizer.item()}, Variance: {torch.mean(decode_var).item()}")
        print(f"ELBO: {elbo.item()}, Reconstruction error: {reconstruction_error.item()}, Regularizer: {regularizer.item()}, Variance: {torch.mean(decode_var).item()}")
        return  (elbo, reconstruction_error, regularizer) if not save_latent else (elbo, reconstruction_error, regularizer, z)

    def initialise(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight, gain=nn.init.calculate_gain('leaky_relu'))
                m.bias.data.fill_(0)

        self.apply(_init_weights)

    def train_VAE(self, dataloader, epochs, lr=10e-5):
        parameters = [
            param for param in self.parameters() if param.requires_grad == True
        ]
        optimizer = torch.optim.Adam(parameters, lr=lr)

        REs = []
        KLs = []
        ELBOs = []

        self.initialise()
        self.train()
        for epoch in tqdm(range(epochs)):
            for batch in dataloader:
                x = batch['image'].to(device)
                optimizer.zero_grad()
                elbo, RE, KL = self.forward(x)
                
                REs, KLs, ELBOs = REs + [RE.item()], KLs + [KL.item()], ELBOs + [elbo.item()]

                elbo.backward()
                optimizer.step()

            tqdm.write(
                f"Epoch: {epoch+1}, ELBO: {elbo.item()}, Reconstruction Error: {RE.item()}, Regularizer: {KL.item()}, Variance: {torch.mean(self.var).item()}"
            )

        return (
            self.encoder,
            self.decoder,
            REs,
            KLs,
            ELBOs
        )
    
    def test_eval(self, dataloader, save_latent=False, results_folder=''):
        # only wors if len of dataloader is divisible by batch_size
        REs, KLs, ELBOs = [], [], []

        moa, compound = [], []

        latent = np.zeros((self.latent_dim, len(dataloader) * dataloader.batch_size)).T
        self.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader)):
                x = batch["image"].to(device)

                moa, compound = moa + batch["moa_name"], compound + batch["compound"]

                if save_latent:
                    elbo, RE, KL, z = self.forward(x, save_latent=save_latent)
                    z = z.cpu().detach().numpy()
                    latent[i*dataloader.batch_size:(i+1)*dataloader.batch_size, :] = z

                else:
                    elbo, RE, KL = self.forward(x)

                REs, KLs, ELBOs = REs + [RE.item()], KLs + [KL.item()], ELBOs + [elbo.item()]

            if save_latent: 
                np.savez(results_folder + "latent_space.npz", z=latent, labels=moa, compound=compound)

            return REs, KLs, ELBOs


def generate_image(X, vae, latent_dim, channels, input_dim, batch_size=1):
    vae.eval()
    X = X.to(device)
    mu, log_var = vae.encode(X)
    eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)
    z = mu + torch.exp(0.5 * log_var) * eps
    mean, var = vae.decode(z)
    image = torch.normal(mean=mean, std=torch.sqrt(var)).to(device)
    image = image.view(channels, input_dim, input_dim)
    image = image.clip(0,1).detach().cpu().numpy()
    return image



def plot_1_reconstruction(image, 
                        vae,
                        name, 
                        latent_dim, 
                        channels, 
                        input_dim, 
                        results_folder=''):
    recon_image = generate_image(
        image,
        vae=vae,
        latent_dim=latent_dim,
        channels=channels,
        input_dim=input_dim,
    )


    fig, ax = plt.subplots(1, 2, figsize=(10,6))

    ax = ax.flatten()
    ax[0].imshow(image.reshape((68,68,3)), cmap="gray")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Original',  size=22,fontweight="bold")
    ax[1].imshow(recon_image.reshape((68,68,3)), cmap="gray")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Reconstruction', size=22,fontweight="bold")
    fig.suptitle(name,  size=26, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_folder + name +'.png')
    plt.show()
    plt.close()



def plot_ELBO(REs, KLs, ELBOs, name, results_folder):
    fig = plt.figure(figsize=(10,6))
    plt.plot(REs, label='Reconstruction Error', color='red', 
             linestyle='--', linewidth=2, alpha=0.5)
    
    plt.plot(KLs, label='Regularizer', color='blue', 
             linestyle='--', linewidth=2, alpha=0.5)
    
    plt.plot(ELBOs, label='ELBO', color='black', 
             linestyle='-', linewidth=4, alpha=0.5)
    plt.legend(fontsize=15)
    plt.title(name, size=26, fontweight='bold')
    plt.xlabel('Iterations',  size=22, fontweight='bold')
    plt.ylabel('Loss',  size=22, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.savefig(results_folder + name + '.png')
    plt.show()
    plt.close()






if __name__ == "__main__":
    latent_dim = 300
    epochs = 100
    batch_size = 100

    input_dim = 68
    channels = 3

    train_size = 100_000
    test_size = 30_000

    #latent_dim = 10
   # epochs, batch_size, train_size, test_size = 2, 10, 10, 2000

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

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


    X_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, drop_last=True)
    X_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)


    VAE = VAE(
        latent_dim=latent_dim,
        input_dim=input_dim,
        channels=channels,
        beta=0.1
    ).to(device)
    

    classes, indexes = np.unique(dataset_test.meta[dataset_test.col_names[-1]], return_index=True)
    boolean = len(classes) == 13 if not exclude_dmso else len(classes) == 12

    if not boolean: 
       raise Warning("The number of unique drugs in the test set is not 13")

    #print("VAE:")
    #summary(VAE, input_size=(channels, input_dim, input_dim))
    results_folder = 'final_vae/'

    if not(os.path.exists(results_folder)):
        os.mkdir(results_folder)


    encoder_VAE, decoder_VAE, train_REs, train_KLs, train_ELBOs = VAE.train_VAE(dataloader=X_train, epochs=epochs)

    test_REs, test_KLs, test_ELBOs = VAE.test_eval(dataloader=X_test, save_latent=True, results_folder=results_folder)

    # np.savez("latent_space_VAE.npz", latent_space=latent_space.detach().numpy())
    

    train_images_folder = results_folder +'train_images/'

    test_images_folder = results_folder + 'test_images/'

    if not(os.path.exists(train_images_folder)):
        os.mkdir(train_images_folder)
    
    if not(os.path.exists(test_images_folder)):
        os.mkdir(test_images_folder)
    
    np.savez(results_folder + "train_ELBOs.npz", train_ELBOs=train_ELBOs)
    np.savez(results_folder + "train_REs.npz", train_REs=train_REs)
    np.savez(results_folder + "train_KLs.npz", train_KLs=train_KLs)
    np.savez(results_folder + "test_ELBOs.npz", test_ELBOs=test_ELBOs)
    np.savez(results_folder + "test_REs.npz", test_REs=test_REs)
    np.savez(results_folder + "test_KLs.npz", test_KLs=test_KLs)


    torch.save(encoder_VAE, results_folder + "encoder.pt")
    torch.save(decoder_VAE, results_folder + "decoder.pt")


    plot_ELBO(train_REs, train_KLs, train_ELBOs, name="ELBO-components", results_folder=results_folder)
        

    for i, image in enumerate(X_train.dataset):
        if i == 10:
            break
        plot_1_reconstruction(image['image'],
                            vae = VAE,
                            name="Train " + str(image['id']), 
                            results_folder=train_images_folder, 
                            latent_dim=latent_dim, 
                            channels=channels, 
                            input_dim=input_dim)
    
    if boolean:
        for index in indexes:
            image = dataset_test[index]
            plot_1_reconstruction(image['image'],
                                vae = VAE,
                                name=image['moa_name'] + ' ' + str(image['id']), 
                                results_folder=test_images_folder, 
                                latent_dim=latent_dim, 
                                channels=channels, 
                                input_dim=input_dim)
            
    
    
    from interpolation import interpolate_between_two_images, interpolate_between_three_images
    
    interpolate_between_two_images(VAE, 452305, 475106, main_path, results_folder=results_folder)
    interpolate_between_three_images(VAE, 452305, 475106, 273028, main_path, results_folder=results_folder)



