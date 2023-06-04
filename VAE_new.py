import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from torchsummary import summary
from tqdm import tqdm
import torchvision


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# detect anomaly
torch.autograd.set_detect_anomaly(True)

# changing the posterior to be a bernoulli distribution 

def log_Categorical(x, theta, num_classes):
    x_one_hot = nn.functional.one_hot(
        x.flatten(start_dim=1, end_dim=-1).long(), num_classes=num_classes)
    log_p = torch.sum(
        x_one_hot * torch.log(theta), dim=-1)
    return log_p


def log_Normal(x, mu, log_var):
    D = x.shape[1]
    log_p = -.5 * ((x - mu) ** 2. * torch.exp(-log_var) +
                   log_var + D * torch.log(2 * torch.tensor(np.pi)))
    return log_p


def log_standard_Normal(x):
    D = x.shape[1]
    log_p = -.5 * (x ** 2. + D * torch.log(2 * torch.tensor(np.pi)))
    return log_p


class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels):
        super(encoder, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(
            32 * self.input_dim * self.input_dim, 2 * latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32 * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        return x


class decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels, pixel_range):
        super(decoder, self).__init__()
        self.input_dim = input_dim
        self.channels = channels
        self.pixel_range = pixel_range

        self.input = nn.Linear(
            latent_dim, 32 * self.input_dim * self.input_dim)
        self.conv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.conv2 = nn.ConvTranspose2d(
            16, channels, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        # conversion to [0,1]
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32, self.input_dim, self.input_dim)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, self.channels, self.input_dim, self.input_dim)
        # x = self.fully_connected(x)
        # x = x.view(-1, self.channels * self.input_dim *
        #            self.input_dim, self.pixel_range)
        # x = self.softmax(x)
        return x


class VAE(nn.Module):
    def __init__(self, X, pixel_range, latent_dim, input_dim, channels):
        super(VAE, self).__init__()
        self.encoder = encoder(input_dim, latent_dim, channels)
        self.decoder = decoder(input_dim, latent_dim, channels, pixel_range)
        self.input_dim = input_dim
        self.pixel_range = pixel_range
        self.latent_dim = latent_dim

        self.data_length = len(X)
        # self.prior = torch.distributions.MultivariateNormal(loc=torch.zeros(latent_dim), covariance_matrix=torch.eye(latent_dim))

    def encode(self, x):
        mu, log_var = torch.split(
            self.encoder.forward(x), self.latent_dim, dim=1)
        return mu, log_var

    def reparameterization(self, mu, log_var):
        eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)
        return mu + torch.exp(0.5*log_var) * eps

    def decode(self, z):
        return self.decoder.forward(z)

    def forward(self, x, reduction='sum'):
        mu, log_var = self.encode(x)


        # KL divergence term
        p = torch.distributions.Normal(torch.zeros(self.latent_dim).to(device), torch.ones(self.latent_dim).to(device))
        q = torch.distributions.Normal(mu, torch.exp(0.5*log_var))

       
        z = self.reparameterization(mu, log_var)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)  

        kl = (log_qzx - log_pz) 

        # sum 
        if reduction == 'sum':
            kl = torch.sum(kl, dim=-1)
        else:
            kl = torch.mean(kl, dim=-1)
        
        # reconstruction error
        # x_tilde is the parameters of the distribution

        x_tilde = self.decode(z)


        # this nexxxt part is unclear to me
        log_scale = torch.nn.parameter.Parameter(torch.tensor([0.0]))
        scale = torch.exp(log_scale)
        dist = torch.distributions.Normal(x_tilde, scale.to(device))

        log_pxz = dist.log_prob(x)
        log_pxz = torch.sum(log_pxz, dim=(1, 2, 3))

        reconstruct = -log_pxz


        elbo = kl + reconstruct
        elbo = elbo.mean()
        

        tqdm.write(
            f"ELBO: {elbo}, Reconstruction error: { reconstruct.mean()}, Regularizer: {kl.mean()}")

        return elbo, reconstruct.mean(), kl.mean()

    def train_VAE(self, dataloader, epochs, batch_size, lr=10e-5):
        parameters = [param for param in self.parameters()
                      if param.requires_grad == True]
        optimizer = torch.optim.Adam(parameters, lr=lr)

        reconstruction_errors = []
        regularizers = []

        ## self.train()
        for epoch in tqdm(range(epochs)):
            for batch in tqdm(dataloader):
                
                x = batch.to(device)
                x = x/255
    
                elbo, reconstruction_error, regularizer = self.forward(x)
                reconstruction_errors.append(
                    reconstruction_error.cpu().detach().numpy())
                regularizers.append(regularizer.cpu().detach().numpy())

                optimizer.zero_grad()
                elbo.backward()
                optimizer.step()

            tqdm.write(
                f"Epoch: {epoch+1}, ELBO: {elbo.detach()}, Reconstruction Error: {reconstruction_error.detach()}, Regularizer: {regularizer.detach()}")

        mu, log_var = self.encode(x)
        latent_space = self.reparameterization(mu, log_var)

        return self.encoder, self.decoder, reconstruction_errors, regularizers, latent_space


def generate_image(X, encoder, decoder, latent_dim, channels, input_dim, batch_size=1):
    encoder.eval()
    decoder.eval()
    X = X.to(device)
    mu, log_var = torch.split(encoder.forward(X), latent_dim, dim=1)
    eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)
    z = mu + torch.exp(0.5*log_var) * eps
    x_tilde = decoder.forward(z)
    ## image = torch.argmax(theta, dim=-1)
     #image = image.reshape((batch_size, channels, input_dim, input_dim))
    # image = torch.permute(image, (0, 2, 3, 1))
    log_scale = torch.nn.parameter.Parameter(torch.tensor([0.0]))
    scale = torch.exp(log_scale)
    dist = torch.distributions.Normal(x_tilde, scale.to(device))
    image = dist.sample()
    return image


latent_dim = 8
epochs = 10
batch_size = 10

pixel_range = 256
input_dim = 28
channels = 1

train_size = 60000
test_size = 10000

learning_rate = 1e-5


trainset = datasets.MNIST(
    root='./MNIST', train=True, download=False, transform=None)
testset = datasets.MNIST(
    root='./MNIST', train=False, download=False, transform=None)
X_train = DataLoader(trainset.data[:train_size].reshape(
    (train_size, channels, input_dim, input_dim)).float(), batch_size=batch_size, shuffle=True)

X_test = DataLoader(testset.data[:test_size].reshape(
                        (test_size, channels, input_dim, input_dim)).float(), 
                        batch_size=batch_size, shuffle=True)

# X = np.load("image_matrix.npz")["images"][:1000]
# X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)

VAE = VAE(X_train, pixel_range=pixel_range,
          latent_dim=latent_dim, input_dim=input_dim, channels=channels).to(device)

# print("VAE:")
# summary(VAE, input_size=(channels, input_dim, input_dim))

encoder_VAE, decoder_VAE, reconstruction_errors, regularizers, latent_space = VAE.train_VAE(
    dataloader=X_train, epochs=epochs, batch_size=batch_size, lr=learning_rate)


save_folder_path = "VAE_results/"


name = "learning_rate_" + str(learning_rate) + "_latent_dim_" + str(latent_dim) + "_epochs_" + str(epochs) + "_batch_size_" + str(batch_size) + "_"

np.savez(save_folder_path + name + "latent_space.npz", latent_space=latent_space.detach().cpu().numpy())
np.savez(save_folder_path + name + "reconstruction_errors.npz", reconstruction_errors=np.array(reconstruction_errors))
np.savez(save_folder_path + name + "regularizers.npz", regularizers=np.array(regularizers))

generated_images = []

for batch in X_test:
    for j in range(batch_size):
        image = generate_image(batch[j], encoder_VAE, decoder=decoder_VAE,
                            latent_dim=latent_dim, channels=channels, input_dim=input_dim)
        generated_images.append(image)
        if len(generated_images) == 9:
            break
    if len(generated_images) == 9:
            break
    

np.savez(save_folder_path + name + "generated_images.npz", generated_images=generated_images.detach().cpu().numpy())

