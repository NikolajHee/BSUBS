
from VAE import log_Categorical, log_Normal, log_standard_Normal, encoder, decoder, VAE, generate_image
import torch
import numpy as np
from torchvision import datasets


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load torch model (encoder.pt, decoder.pt)

latent_dim = 2
epochs = 10
batch_size = 30

pixel_range = 256
input_dim = 28
channels = 1

testset = datasets.MNIST(
    root='./MNIST', train=False, download=True, transform=None)

labels = testset.targets.numpy()

model = encoder(latent_dim=latent_dim, input_dim=input_dim, channels=channels).to(device)

model.load_state_dict(torch.load("encoder.pt"))

# save latent space
latent_points = np.zeros((len(testset), latent_dim))
label = np.zeros((len(testset)))

for i, image in enumerate(testset.data):
    image = image.reshape((1, channels, input_dim, input_dim)).float()
    latent_point = model.forward(image)
    latent_points[i] = latent_point.detach().cpu().numpy()
    label[i] = labels[i]

np.save("latent_space", latent_points)
np.save("labels", label)