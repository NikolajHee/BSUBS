import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#from torchsummary import summary
from tqdm import tqdm
import os
from torch.functional import F




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sumlogC( x , eps = 1e-5):
    '''
    Numerically stable implementation of 
    sum of logarithm of Continous Bernoulli
    constant C, using Taylor 2nd degree approximation
        
    Parameter
    ----------
    x : Tensor of dimensions (batch_size, dim)
        x takes values in (0,1)
    ''' 
    x = torch.clamp(x, eps, 1.-eps) 
    mask = torch.abs(x - 0.5).ge(eps)
    far = torch.masked_select(x, mask)
    close = torch.masked_select(x, ~mask)
    far_values =  torch.log( (torch.log(1. - far) - torch.log(far)).div(1. - 2. * far) )
    close_values = torch.log(torch.tensor((2.))) + torch.log(1. + torch.pow( 1. - 2. * close, 2)/3. )
    return far_values.sum() + close_values.sum()


def loss_cbvae(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 68*68*3), reduction='none').sum(-1).mean()
    KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),1)).mean()
    LOGC = torch.tensor([sumlogC(xx) for xx in recon_x.view(-1, 68*68*3)]).mean()
    print('BCE: ', BCE)
    print('KLD: ', KLD)
    print('LOGC: ', LOGC)
    return BCE + KLD + LOGC




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
                nn.Conv2d(channels, out_channels=h, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h),
                nn.LeakyReLU(leaky_relu_slope)
            ))
            channels = h
        
        self.encoder = nn.Sequential(*save)
        # final layer
        self.to_latent = nn.Sequential(nn.Linear(self.hidden_channels[-1]*9*9, latent_dim * 2),
                                       nn.LeakyReLU(leaky_relu_slope))

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
        self.from_latent = nn.Linear(latent_dim, hidden_channels[0]*9*9)

        # loop over hidden channels
        save = []

        for i in range(len(hidden_channels) -1):
            save.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=hidden_channels[i], out_channels=hidden_channels[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_channels[i+1]),
                nn.LeakyReLU(leaky_relu_slope)
            ))
        
        self.decoder = nn.Sequential(*save)

        # final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_channels[-1], out_channels=channels, kernel_size=3, stride=2, padding=3, output_padding=1),
            nn.Flatten(start_dim=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.from_latent(x)
        x = x.view(-1, self.hidden_channels[0], 9, 9)
        x = self.decoder(x)
        x = self.final(x)
        return x

BATCH_SIZE = 100

class VAE(nn.Module):
    def __init__(self, latent_dim, input_dim, channels, batch_size, hidden_channels: list=[8, 16, 32]):
        super(VAE, self).__init__()
        self.batch_size = batch_size
        self.encoder = encoder(input_dim, latent_dim, channels, hidden_channels)
        self.decoder = decoder(input_dim, latent_dim, channels, [i for i in reversed(hidden_channels)])
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
        return self.decoder(z)

    def forward(self, x, save_latent = False):
        mu, log_var = self.encode(x)
        x = x.view(-1, self.input_dim * self.input_dim * self.channels)
        z = self.reparameterization(mu, log_var)

        recon_x = self.decode(z)

        loss = loss_cbvae(recon_x, x, mu, log_var)

        tqdm.write(
            f"Loss: {loss.item():.3f} ")

        return loss

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

    def train_VAE(self, dataloader, epochs, lr=1e-3):
        parameters = [
            param for param in self.parameters() if param.requires_grad == True
        ]
        optimizer = torch.optim.Adam(parameters, lr=lr)

        loss_list = []

        # self.initialise()
        self.train()
        for epoch in tqdm(range(epochs)):
            for batch in tqdm(dataloader):
                x = batch['image'].to(device)
                optimizer.zero_grad()
                loss = self.forward(x)
                
                loss_list = loss_list + [loss.item()]

                loss.backward()
                optimizer.step()

        return (
            loss_list,
        )


def generate_image(X, vae, latent_dim, channels, input_dim, batch_size=1):
    vae.eval()
    X = X.to(device)
    mu, log_var = vae.encode(X.view(batch_size, channels, input_dim, input_dim))
    eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)
    z = mu + torch.exp(0.5 * log_var) * eps
    image = vae.decode(z)
    #image = torch.normal(mean=mean, std=torch.sqrt(var)).to(device)
    #image = image.view(channels, input_dim, input_dim)
    image = image.clip(0,1).detach().cpu().numpy()
    return image

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

if __name__ == "__main__":
    latent_dim = 200
    epochs = 10
    batch_size = 100

    input_dim = 68
    channels = 3

    train_size = 10000
    test_size = 1000

    #epochs, batch_size, train_size = 2, 1, 100

    # torch.backends.cudnn.deterministic = True
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)

    from dataloader import BBBC

    main_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"
    main_path = "/Users/nikolaj/Fagprojekt/Data/"


    exclude_dmso = False
    shuffle = True

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


    X_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    X_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    VAE = VAE(
        batch_size=batch_size,
        latent_dim=latent_dim,
        input_dim=input_dim,
        channels=channels,
        hidden_channels=[8,16,32]
    ).to(device)

    #print("VAE:")
    #summary(VAE, input_size=(channels, input_dim, input_dim))

    loss = VAE.train_VAE(
        dataloader=X_train, epochs=epochs)



    # np.savez("latent_space_VAE.npz", latent_space=latent_space.detach().numpy())



    results_folder = 'beta/'
    if not(os.path.exists(results_folder)):
        os.mkdir(results_folder)



    x = np.arange(0, len(loss), 1)

    plt.plot(x, loss, label="loss")
    plt.xlabel("iterationns")
    plt.title("ELBO Components")
    plt.legend()
    plt.savefig(results_folder + 'ELBO_components.png')
    plt.show()
        
        

    plot_reconstruction(X_train.dataset, 
                        vae = VAE,
                        name="Training_images_reconstructed", 
                        results_folder=results_folder, 
                        latent_dim=latent_dim, 
                        channels=channels, 
                        input_dim=input_dim)
    

    plot_reconstruction(X_test.dataset, 
                        vae = VAE,
                        name="Test_images_reconstructed", 
                        results_folder=results_folder, 
                        latent_dim=latent_dim, 
                        channels=channels, 
                        input_dim=input_dim)


