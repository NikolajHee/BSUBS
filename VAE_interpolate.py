import torch
from VAE import log_Categorical, log_Normal, log_standard_Normal, encoder, decoder, VAE, generate_image
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# gridsearch for latent_dim

save_folder_path = "interpol_results/" # 4, 8, 16, 32, 64, 128

latent_dim = 16
epochs = 10
batch_size = 30

pixel_range = 256
input_dim = 28
channels = 1


trainset = datasets.MNIST(
    root='./MNIST', train=True, download=True, transform=None)
testset = datasets.MNIST(
    root='./MNIST', train=False, download=True, transform=None)
X_train = trainset.data.reshape(
    (len(trainset), channels, input_dim, input_dim)).float()
X_test = testset.data.reshape(
    (len(testset), channels, input_dim, input_dim)).float()

# subsetting train set
#X_train = X_train[:10]



print("latent_dim: ", latent_dim)

VAE_ = VAE(X_train, pixel_range=pixel_range,
        latent_dim=latent_dim, input_dim=input_dim, channels=channels).to(device)
print("VAE_ created")
encoder_VAE, decoder_VAE, reconstruction_errors, regularizers, latent_space, error_log = VAE_.train_VAE(
    X=X_train, epochs=epochs, batch_size=batch_size)
print("VAE trained")

name = "latent_dim_" + str(latent_dim) + "_epochs_" + str(epochs) + "_batch_size_" + str(batch_size) + "_"

# np.savez(save_folder_path + name + "latent_space.npz", latent_space=latent_space.detach().cpu().numpy())
# np.savez(save_folder_path + name + "reconstruction_errors.npz", reconstruction_errors=np.array(reconstruction_errors))
# np.savez(save_folder_path + name + "regularizers.npz", regularizers=np.array(regularizers))

# f = open(save_folder_path + name + "error_log.txt", "w")
# f.write( str(error_log))
# f.close()


# generated_images = []
# for i in range(9):
#     image = generate_image(X_test[i], encoder_VAE, decoder=decoder_VAE,
#                         latent_dim=latent_dim, channels=channels, input_dim=input_dim)
#     generated_images.append(image)

# np.savez(save_folder_path + name + "generated_images.npz", generated_images=np.array(generated_images))


#**** Interpolation

def interpolate(v1, v2, Nstep):
    for i in range(Nstep):
        r = v2 - v1
        v = v1 + r * (i / (Nstep - 1))
        yield v

X_3 = X_test.data[0] 
# testset.data[testset.targets == 3]
X_5 = X_test.data[1] 
# testset.data[testset.targets == 5]


def _encode(x, model):
    mu, log_var = model.encode(x)
    z = model.reparameterization(mu, log_var)
    return z

z1 = _encode(X_3, VAE_)
z2 = _encode(X_5, VAE_)

interpolated_images = []

for z in interpolate(z1, z2, 10):
    theta = VAE_.decode(z)
    image = torch.argmax(theta, dim=-1)
    image = image.reshape((1, channels, input_dim, input_dim))
    image = torch.permute(image, (0, 2, 3, 1))
    image = image.cpu().numpy()

    interpolated_images.append(image)


np.savez(save_folder_path + name + "interpolated_images.npz", interpolated_images=np.array(interpolated_images))


