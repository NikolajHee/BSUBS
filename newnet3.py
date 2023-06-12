import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#from torchsummary import summary
from tqdm import tqdm
import os
import sys


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
        x = nn.LeakyReLU(0.01)(x)
        return x


class decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels):
        super(decoder, self).__init__()
        self.input_dim = input_dim
        self.channels = channels

        self.input = nn.Linear(
            latent_dim,  2 * 16 * self.input_dim * self.input_dim)
        self.conv2 = nn.ConvTranspose2d(
            16, channels, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        #self.output = nn.Linear(channels * self.input_dim * self.input_dim,
        #                        2 * channels * self.input_dim * self.input_dim)

        # self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = nn.LeakyReLU(0.01)(x)
        x = x.view(-1, 16, self.input_dim, self.input_dim)
        x = self.conv2(x)
        x = nn.LeakyReLU(0.01)(x)
        x = x.view(-1,  2 * self.channels * self.input_dim * self.input_dim)
        #x = self.output(x)
        #x = nn.LeakyReLU(0.01)(x)
        # x = self.softmax(x) # i dont think this is needed.. but maybe?
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim, input_dim, channels, beta = 1):
        super(VAE, self).__init__()
        self.encoder = encoder(input_dim, latent_dim, channels)
        self.decoder = decoder(input_dim, latent_dim, channels)
        self.latent_dim = latent_dim
        self.channels = channels
        self.input_dim = input_dim
        self.beta = beta

        # self.prior = torch.distributions.MultivariateNormal(loc=torch.zeros(latent_dim), covariance_matrix=torch.eye(latent_dim))

    def encode(self, x):
        mu, log_var = torch.split(
            self.encoder.forward(x), self.latent_dim, dim=1)
        return mu, log_var

    def reparameterization(self, mu, log_var):
        self.eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)
        return mu + torch.exp(0.5 * log_var) * self.eps

    def decode(self, z):
        mu, log_var = torch.split(
            self.decoder.forward(z), self.channels * self.input_dim * self.input_dim, dim=1)
        std = torch.exp(log_var)
        return mu, std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterization(mu, log_var)

        decode_mu, decode_var = self.decode(z)
        # decode_std = 0.1 * torch.ones(decode_mu.shape).to(device)
        log_posterior = log_Normal(z, mu, log_var)
        log_prior = log_standard_Normal(z)

        print(decode_var)
        log_like = (1 / (2 * (decode_var)) * nn.functional.mse_loss(decode_mu, x.flatten(
            start_dim=1, end_dim=-1), reduction="none")) + 0.5 * torch.log(decode_var) + 0.5 * torch.log(2 * torch.tensor(np.pi))
        #print(decode_var)

        reconstruction_error = torch.sum(log_like, dim=-1).mean()
        regularizer = - torch.sum(log_prior - log_posterior, dim=-1).mean() 

        elbo = reconstruction_error + regularizer * self.beta

        tqdm.write(
            f"ELBO: {elbo.item()}, Reconstruction error: {reconstruction_error.item()}, Regularizer: {regularizer.item()}")

        return elbo, reconstruction_error, regularizer

    def initialise(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.sparse_(m.weight, sparsity=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(3)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.dirac_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(_init_weights)

    def train_VAE(self, dataloader, epochs, lr=10e-5):
        parameters = [
            param for param in self.parameters() if param.requires_grad == True
        ]
        optimizer = torch.optim.Adam(parameters, lr=lr)

        reconstruction_errors = []
        regularizers = []

        # self.initialise()
        self.train()
        for epoch in tqdm(range(epochs)):
            for batch in tqdm(dataloader):
                x = batch['image'].to(device)
                optimizer.zero_grad()
                elbo, reconstruction_error, regularizer = self.forward(x)
                reconstruction_errors.append(
                    reconstruction_error.item())
                regularizers.append(regularizer.item())
                elbo.backward()
                optimizer.step()

            tqdm.write(
                f"Epoch: {epoch+1}, ELBO: {elbo.item()}, Reconstruction Error: {reconstruction_error.item()}, Regularizer: {regularizer.item()}"
            )

        mu, log_var = self.encode(x)
        latent_space = self.reparameterization(mu, log_var)

        return (
            self.encoder,
            self.decoder,
            reconstruction_errors,
            regularizers,
            latent_space,
        )


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


latent_dim = 200
epochs = 100
batch_size = 40

input_dim = 68
channels = 3

train_size = 20000
test_size = 1000

# epochs, batch_size, train_size = 2, 1, 100

# torch.backends.cudnn.deterministic = True
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

from dataloader import BBBC

main_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"
#main_path = "/Users/nikolaj/Fagprojekt/Data/"

exclude_dmso = False
shuffle = False

subset = (train_size, test_size)

dataset_train = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                        meta_path=main_path + "metadata.csv",
                        subset=subset,
                        test=False,
                        exclude_dmso=exclude_dmso,
                        shuffle=False)

dataset_test = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                        meta_path=main_path + "metadata.csv",
                        subset=subset,
                        test=True,
                        exclude_dmso=exclude_dmso,
                        shuffle=False)


X_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
X_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

VAE = VAE(
    latent_dim=latent_dim,
    input_dim=input_dim,
    channels=channels,
    beta=0.1,
).to(device)

#print("VAE:")
##summary(VAE, input_size=(channels, input_dim, input_dim))

encoder_VAE, decoder_VAE, reconstruction_errors, regularizers, latent_space = VAE.train_VAE(
    dataloader=X_train, epochs=epochs)

# torch.save(encoder_VAE, "encoder_VAE.pt")
# torch.save(decoder_VAE, "decoder_VAE.pt")

# np.savez("latent_space_VAE.npz", latent_space=latent_space.detach().numpy())



results_folder = str(0.1) + 'gaussian/'
if not(os.path.exists(results_folder)):
    os.mkdir(results_folder)

np.savez(results_folder + "latent_space_VAE.npz", latent_space=latent_space.cpu().detach().numpy())
np.savez(results_folder + "reconstruction_errors_VAE.npz", reconstruction_errors=reconstruction_errors)


plt.plot(
    np.arange(0, len(reconstruction_errors), 1),
    reconstruction_errors,
    label="Reconstruction Error",
)
plt.plot(np.arange(0, len(regularizers), 1), regularizers, label="Regularizer")
plt.xlabel("iterationns")
plt.title("ELBO Components")
plt.legend()
plt.savefig(results_folder + 'ELBO_components.png')
plt.show()


def plot_reconstruction(data, name, results_folder=''):
    generated_images = []
    for i in range(9):
        image = generate_image(
            data[i]['image'],
            vae=VAE,
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
    
    

plot_reconstruction(X_train.dataset, "Training_images_reconstructed", results_folder=results_folder)
plot_reconstruction(X_test.dataset, "Test_images_reconstructed", results_folder=results_folder)


