import numpy as np



filename = "L=" + str(350) + 

file = "L\=350.\grid/test_ELBOs.npz"
data = np.load(file)
test_ELBOs = data['test_ELBOs']
print(test_ELBOs.mean())