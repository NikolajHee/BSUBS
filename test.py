import numpy as np
import matplotlib.pyplot as plt

folder = '100/'

ELBOs =np.load(folder + "train_ELBOs.npz")['train_ELBOs']
KLs =    np.load(folder + "train_KLs.npz")['train_KLs']
REs =    np.load(folder + "train_REs.npz")['train_REs']


plt.plot(ELBOs, label='ELBO')
plt.plot(KLs, label='KL')
plt.plot(REs, label='RE')
plt.legend()
plt.show()
