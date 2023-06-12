import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from torchsummary import summary
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
    def __init__(self, input_dim, latent_dim, channels, hidden_channels, leaky_relu_slope=0.01):
        super(encoder, self).__init__()
        #* encoder layers
        self.hidden_channels = hidden_channels

        # loop over hidden channels
        save = []
        for h in self.hidden_channels:
            save.append(nn.Sequential(
                nn.Conv2d(channels, out_channels=h, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(h),
                nn.LeakyReLU(leaky_relu_slope)
            ))
            channels = h
        
        self.encoder = nn.Sequential(*save)
        # final layer
        self.to_latent = nn.Sequential(nn.Linear(self.hidden_channels[-1]*68*68, latent_dim * 2))

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.hidden_channels[-1] * x.shape[-1] * x.shape[-1])
        x = self.to_latent(x)
        return x


class decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels, hidden_channels, leaky_relu_slope=0.01):
        super(decoder, self).__init__()

        self.channnels = channels
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels


        #* decoder layers
        self.from_latent = nn.Linear(latent_dim, hidden_channels[0]*68*68)

        # loop over hidden channels
        save = []

        for i in range(len(hidden_channels) - 1):
            save.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=hidden_channels[i], out_channels=hidden_channels[i+1], kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(hidden_channels[i+1]),
                nn.LeakyReLU(leaky_relu_slope)
            ))
        
        self.decoder = nn.Sequential(*save)

        # final layer
        self.final = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(channels * input_dim * input_dim, 2 * channels * input_dim * input_dim)
        )

    def forward(self, x):
        x = self.from_latent(x)
        x = x.view(-1, self.hidden_channels[0], 68, 68)
        x = self.decoder(x)
        x = self.final(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim, input_dim, channels, hidden_channels: list=[8, 16, 32]):
        super(VAE, self).__init__()
        self.encoder = encoder(input_dim, latent_dim, channels, hidden_channels)
        self.decoder = decoder(input_dim, latent_dim, channels, [i for i in reversed(hidden_channels)] + [3])
        self.latent_dim = latent_dim
        self.channels = channels
        self.input_dim = input_dim

        # self.prior = torch.distributions.MultivariateNormal(loc=torch.zeros(latent_dim), covariance_matrix=torch.eye(latent_dim))

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
        var = torch.exp(log_var)
        return mu, var

    def forward(self, x, beta = 0.1, save_latent = False):
        mu, log_var = self.encode(x)
        z = self.reparameterization(mu, log_var)

        decode_mu, decode_var = self.decode(z)
        # decode_std = 0.1 * torch.ones(decode_mu.shape).to(device)
        log_posterior = log_Normal(z, mu, log_var)
        log_prior = log_standard_Normal(z)


        log_like = (1 / (2 * (decode_var)) * nn.functional.mse_loss(decode_mu, x.flatten(
            start_dim=1, end_dim=-1), reduction="none")) + torch.log(torch.sqrt(decode_var)) + 0.5 * torch.log(2 * torch.tensor(np.pi))
        #print(decode_var)

        reconstruction_error = torch.sum(log_like, dim=-1).mean()
        regularizer = - torch.sum(log_prior - log_posterior, dim=-1).mean() 

        elbo = reconstruction_error + regularizer * beta

        tqdm.write(
            f"ELBO: {elbo.item()}, Reconstruction error: {reconstruction_error.item()}, Regularizer: {regularizer.item()}")

        return (elbo, reconstruction_error, regularizer) if not save_latent else (elbo, reconstruction_error, regularizer, z)

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
            for batch in tqdm(dataloader):
                x = batch['image'].to(device)
                optimizer.zero_grad()
                elbo, RE, KL = self.forward(x)
                
                REs, KLs, ELBOs = REs + [RE.item()], KLs + [KL.item()], ELBOs + [elbo.item()]

                elbo.backward()
                optimizer.step()

            tqdm.write(
                f"Epoch: {epoch+1}, ELBO: {elbo.item()}, Reconstruction Error: {RE.item()}, Regularizer: {KL.item()}"
            )

        return (
            self.encoder,
            self.decoder,
            REs,
            KLs,
            ELBOs
        )


def generate_image(X, vae, latent_dim, channels, input_dim, batch_size=1):
    vae.eval()
    X = X.to(device)
    _, _, _, z = vae(X.view(batch_size, channels, input_dim, input_dim), save_latent=True)
    mean, var = vae.decode(z)
    image = torch.normal(mean=mean, std=torch.sqrt(var)).to(device)
    image = image.view(channels, input_dim, input_dim)
    image = image.clip(0,1).detach().cpu().numpy()
    return (image, mean.clip(0,1).detach().cpu().numpy())

def plot_reconstruction(data, 
                        vae,
                        name, 
                        latent_dim, 
                        channels, 
                        input_dim, 
                        results_folder='',):
    generated_images = []
    for i in range(9):
        image = generate_image(
            data[i]['image'],
            vae=vae,
            latent_dim=latent_dim,
            channels=channels,
            input_dim=input_dim,
        )
        generated_images.append(image)

    fig, ax = plt.subplots(9, 2)
    for i in range(9):
        ax[i, 0].imshow(data[i]['image'].reshape((68,68,3)), cmap="gray")
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        ax[i, 1].imshow(generated_images[i].reshape((68,68,3)), cmap="gray")
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
    fig.suptitle(name)
    plt.savefig(results_folder + name + '.png')
    plt.show()


def plot_1_reconstruction(image, 
                        vae,
                        name, 
                        latent_dim, 
                        channels, 
                        input_dim, 
                        results_folder='',
                        plot_mean=True,):
    (recon_image,mean) = generate_image(
        image,
        vae=vae,
        latent_dim=latent_dim,
        channels=channels,
        input_dim=input_dim,
    )
    if plot_mean: 
        fig, ax = plt.subplots(1, 3, figsize=(10,6))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(10,6))

    ax = ax.flatten()
    ax[0].imshow(image.reshape((68,68,3)), cmap="gray")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Original', fontname="Times New Roman", size=22,fontweight="bold")
    ax[1].imshow(recon_image.reshape((68,68,3)), cmap="gray")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Reconstruction', fontname="Times New Roman", size=22,fontweight="bold")
    if plot_mean:
        ax[2].imshow(mean.reshape((68,68,3)), cmap="gray")
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[2].set_title('Mean')
    fig.suptitle(name, fontname='Times New Roman', size=26, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_folder + name +'.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    latent_dim = 200
    epochs = 100
    batch_size = 100

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


    X_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    X_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    VAE = VAE(
        latent_dim=latent_dim,
        input_dim=input_dim,
        channels=channels,
        hidden_channels=[8,16,32]
    ).to(device)
    

    classes, indexes = np.unique(dataset_test.meta[dataset_test.col_names[-1]], return_index=True)
    boolean = len(classes) == 13 if not exclude_dmso else len(classes) == 12

    if not boolean: 
        raise Warning("The number of unique drugs in the test set is not 13")

    # print("VAE:")
    # summary(VAE, input_size=(channels, input_dim, input_dim))

    encoder_VAE, decoder_VAE, REs, KLs, ELBOs = VAE.train_VAE(
        dataloader=X_train, epochs=epochs)


    # np.savez("latent_space_VAE.npz", latent_space=latent_space.detach().numpy())

    results_folder = 'new_net1/'
    if not(os.path.exists(results_folder)):
        os.mkdir(results_folder)

    train_images_folder = results_folder +'train_images/'

    test_images_folder = results_folder + 'test_images/'

    if not(os.path.exists(train_images_folder)):
        os.mkdir(train_images_folder)
    
    if not(os.path.exists(test_images_folder)):
        os.mkdir(test_images_folder)



    torch.save(encoder_VAE, results_folder + "encoder.pt")
    torch.save(decoder_VAE, results_folder + "decoder.pt")


    x = np.arange(0, len(REs), 1)

    plt.plot(x, REs, label="Reconstruction Error")
    plt.plot(x, KLs, label="Regularizer")
    plt.plot(x, ELBOs, label="ELBO")
    plt.xlabel("iterations")
    plt.title("ELBO Components")
    plt.legend()
    plt.savefig(results_folder + 'ELBO_components.png')
    plt.show()
    # save memory
    plt.close()
        


    for i, image in enumerate(X_train.dataset):
        if i == 10:
            break
        plot_1_reconstruction(image['image'],
                            vae = VAE,
                            name="Train " + str(image['id']), 
                            results_folder=train_images_folder, 
                            latent_dim=latent_dim, 
                            channels=channels, 
                            input_dim=input_dim,
                            plot_mean=False)
    
    if boolean:
        for index in indexes:
            image = dataset_test[index]
            plot_1_reconstruction(image['image'],
                                vae = VAE,
                                name=image['moa'] + ' ' + str(image['id']), 
                                results_folder=test_images_folder, 
                                latent_dim=latent_dim, 
                                channels=channels, 
                                input_dim=input_dim,
                                plot_mean=False)
    
