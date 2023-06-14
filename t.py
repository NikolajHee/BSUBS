import numpy as np

def label_encoder(x):
    classes = np.array(['DMSO', 'Actin disruptors', 'Aurora kinase inhibitors',
    'Cholesterol-lowering', 'DNA damage', 'DNA replication',
    'Eg5 inhibitors', 'Epithelial', 'Kinase inhibitors',
    'Microtubule destabilizers', 'Microtubule stabilizers',
    'Protein degradation', 'Protein synthesis'])
    
    return np.where(classes == x)[0][0]
    
def label_decoder(x):
    classes = np.array(['DMSO', 'Actin disruptors', 'Aurora kinase inhibitors',
    'Cholesterol-lowering', 'DNA damage', 'DNA replication',
    'Eg5 inhibitors', 'Epithelial', 'Kinase inhibitors',
    'Microtubule destabilizers', 'Microtubule stabilizers',
    'Protein degradation', 'Protein synthesis'])
    
    return classes[x]



test = np.array(['DMSO', 'Actin disruptors', 'Aurora kinase inhibitors'])

print(label_encoder(test))