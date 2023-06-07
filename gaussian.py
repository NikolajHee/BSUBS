import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
from torch.nn import functional as F
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_Normal(x, mu, log_var):
    D = x.shape[1]
    log_p = -0.5 * (
        (x - mu) ** 2.0 * torch.exp(-log_var)
        + log_var
        + D * torch.log(2 * torch.tensor(np.pi))
    )
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
        self.fully_connected = nn.Linear(16 * self.input_dim * self.input_dim, 2 * latent_dim)

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

        self.input = nn.Linear(latent_dim, 16 * self.input_dim * self.input_dim)
        self.conv2 = nn.ConvTranspose2d(
            16, channels, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        
        #self.softmax = nn.Softmax(dim=1)  
        #self.softmax = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = nn.LeakyReLU(0.01)(x)
        x = x.view(-1, 16, self.input_dim, self.input_dim)
        x = self.conv2(x)
        x = nn.LeakyReLU(0.01)(x)
        x = x.view(-1, self.channels * self.input_dim * self.input_dim)
        #x = self.softmax(x) # i dont think this is needed.. but maybe?
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim, input_dim, channels):
        super(VAE, self).__init__()
        self.encoder = encoder(input_dim, latent_dim, channels)
        self.decoder = decoder(input_dim, latent_dim, channels)
        self.latent_dim = latent_dim
        self.channels = channels
        self.input_dim = input_dim

    def encode(self, x):
        mu, log_var = torch.split(self.encoder.forward(x), self.latent_dim, dim=1)
        return mu, log_var

    def reparameterization(self, mu, log_var):
        eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)
        return mu + torch.exp(0.5 * log_var) * eps

    def decode(self, z):
        return self.decoder.forward(z)

    def forward(self, x, print_=False):
        mu, log_var = self.encode(x/255)
        z = self.reparameterization(mu, log_var)

        recon_x = self.decode(z)

        log_posterior = log_Normal(z, mu, log_var)
        log_prior = log_standard_Normal(z)

        # initial try
        # log_like = torch.normal(mean = mean, std = 0.001) 
        # log_like = torch.distributions.LogNormal(loc=mean, scale=0.01)
        # recon_x = log_like.sample()
        
        # this works? like some papers say to do this
        reconstruction_error = F.mse_loss(recon_x, x.view(-1, self.channels * self.input_dim * self.input_dim)/255, reduction="none").sum(-1)
        reconstruction_error = reconstruction_error.to(device)

        regularizer = -torch.sum(log_prior - log_posterior, dim=-1) 
        ## regularizer  = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1) # alternative
        
        elbo = (reconstruction_error + regularizer).mean()
        

        # print(f"ELBO: {elbo.detach()}, Reconstruction error: {reconstruction_error.detach().mean()}, Regularizer: {regularizer.detach().mean()}")
        if print_:
            tqdm.write(
                f"ELBO: {elbo.detach()}, Reconstruction error: {reconstruction_error.detach().mean()}, Regularizer: {regularizer.detach().mean()}")
            
        return elbo, reconstruction_error.mean(), regularizer.mean()

    def initialise(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.sparse_(m.weight, sparsity=1)
                if m.bias is not None:
                    m.bias.data.fill_(3)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.dirac_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(_init_weights)

    def train_VAE(self, dataloader, epochs, lr=1e-5):
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
                x = batch.to(device)
                optimizer.zero_grad()
                elbo, reconstruction_error, regularizer = self.forward(x)
                reconstruction_errors.append(reconstruction_error.cpu().detach().numpy())
                regularizers.append(regularizer.detach().numpy())
                elbo.backward(retain_graph=True)
                optimizer.step()

            tqdm.write(
                f"Epoch: {epoch+1}, ELBO: {elbo.detach()}, Reconstruction Error: {reconstruction_error.detach()}, Regularizer: {regularizer.detach()}"
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


def generate_image(X, encoder, decoder, latent_dim, channels, input_dim, batch_size=1):
    encoder.eval()
    decoder.eval()
    X = X.to(device)
    mu, log_var = torch.split(encoder.forward(X), latent_dim, dim=1)
    eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)
    z = mu + torch.exp(0.5 * log_var) * eps
    mean = decoder.forward(z)


    log_like = torch.normal(mean = mean, std = 0.1).to(device)
    recon_x = log_like
    image = recon_x.reshape((channels, input_dim, input_dim))
    image = image.cpu().detach().numpy()
    return image


latent_dim = 4
epochs = 100
batch_size = 20

input_dim = 28
channels = 1

train_size = 10000
test_size = 1000

#torch.backends.cudnn.deterministic = True
#torch.manual_seed(42)
#∑torch.cuda.manual_seed(42)

trainset = datasets.MNIST(root="./MNIST", train=True, download=True, transform=None)
testset = datasets.MNIST(root="./MNIST", train=False, download=True, transform=None)
X_train = DataLoader(
    trainset.data[:train_size]
    .reshape((train_size, channels, input_dim, input_dim))
    .float(),
    batch_size=batch_size,
    shuffle=True,
)
X_test = DataLoader(
    testset.data[:test_size]
    .reshape((test_size, channels, input_dim, input_dim))
    .float(),
    batch_size=batch_size,
    shuffle=True,
)
"""
X = np.load("image_matrix.npz")["images"][:1000]
X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
X_train = DataLoader(X, batch_size=batch_size, shuffle=True)
X_test = DataLoader(X, batch_size=batch_size, shuffle=True)
"""
VAE = VAE(
    latent_dim=latent_dim,
    input_dim=input_dim,
    channels=channels,
).to(device)

print("VAE:")
summary(VAE, input_size=(channels, input_dim, input_dim))

encoder_VAE, decoder_VAE, reconstruction_errors, regularizers, latent_space = VAE.train_VAE(
    dataloader=X_train, epochs=epochs)

results_folder = 'results/'
if not(os.path.exists(results_folder)):
    os.mkdir(results_folder)


torch.save(encoder_VAE, results_folder + "encoder_VAE.pt")
torch.save(decoder_VAE, results_folder + "decoder_VAE.pt")

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
            data[i],
            encoder_VAE,
            decoder=decoder_VAE,
            latent_dim=latent_dim,
            channels=channels,
            input_dim=input_dim,
        )
        generated_images.append(image)

    fig, ax = plt.subplots(9, 2)
    for i in range(9):
        ax[i, 0].imshow(data[i].reshape((28,28,1)), cmap="gray")
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        ax[i, 1].imshow(generated_images[i].reshape((28,28,1)), cmap="gray")
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
    fig.suptitle(name)
    plt.savefig(results_folder + name + '.png')
    plt.show()
    
    

plot_reconstruction(X_train.dataset[:9], "Training_images_reconstructed", results_folder=results_folder)
plot_reconstruction(X_test.dataset[:9], "Test_images_reconstructed", results_folder=results_folder)

