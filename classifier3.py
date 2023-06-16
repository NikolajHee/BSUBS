import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import scipy.stats as st
# random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from statsmodels.stats.contingency_tables import mcnemar

#load data
X_basic = np.load("latent_space_vanilla.npz")["z"]
y = np.load("latent_space_vanilla.npz")["labels"]
compound = np.load("latent_space_vanilla.npz")["compound"]
X_semi = np.load("latent_space_semi.npz")["z"]


index = np.where(y != 'DMSO')


y = y[index]

compound = compound[index]
X_basic=X_basic[index]

def label_encoder(x):
    classes = np.array([ 'Actin disruptors', 'Aurora kinase inhibitors',
    'Cholesterol-lowering', 'DNA damage', 'DNA replication',
    'Eg5 inhibitors', 'Epithelial', 'Kinase inhibitors',
    'Microtubule destabilizers', 'Microtubule stabilizers',
    'Protein degradation', 'Protein synthesis'])
    
    return np.where(classes == x)[0][0]

for i in range(len(y)):
    y[i]=label_encoder(y[i])
y=y.astype('int16')
#get unique compunds
list_of_compounds = np.unique(compound)

#define SVC
model_basic = RandomForestClassifier()
model_semi = RandomForestClassifier()

#Define lists for 
score_basic = []
score_semi = []

pred_basic = np.array([0])
pred_semi = np.array([0])

num_data=[]

real_label = np.array([0])

# Define feature sensitivities
feature_sensitivity_basic = []
feature_sensitivity_semi = []

for Unique_Compound in list_of_compounds:
    print(Unique_Compound)
    #find compound
    test_comp = np.where(compound == Unique_Compound)
    train_comp = np.where(compound != Unique_Compound)

    #index labels
    y_test = y[test_comp]
    y_train = y[train_comp]

    num_data.append(len(y_test)/len(y))

    #index basic VAE train and test
    X_basic_test = X_basic[test_comp]
    X_basic_train = X_basic[train_comp]

    #fit models
    model_basic.fit(X_basic_train, y_train)
    predict_basic=model_basic.predict(X_basic_test)
    pred_basic = np.hstack((pred_basic, predict_basic))

    print(pred_basic.shape)

    #append scores and test labels
    score_basic.append(model_basic.score(X_basic_test, y_test)) 
    real_label = np.hstack((real_label,y_test))  
    print(real_label.shape)

    #sensitivity 
    result_basic = permutation_importance(model_basic, X_basic_test, y_test, n_repeats=10, random_state=0)
    importance_basic = result_basic.importances_mean
    feature_sensitivity_basic.append(importance_basic)

    #index semi VAE train and test
    X_semi_train = X_semi[train_comp]
    X_semi_test = X_semi[test_comp]

    #fit models
    model_semi.fit(X_semi_train, y_train)
    predict_semi = (model_semi.predict(X_semi_test))
    pred_semi = np.hstack((pred_semi,predict_semi))

    #append scores and test labels
    score_semi.append(model_semi.score(X_semi_test, y_test))

    #sensitivity 
    result_semi = permutation_importance(model_semi, X_semi_test, y_test, n_repeats=10, random_state=0)
    importance_semi = result_semi.importances_mean
    feature_sensitivity_semi.append(importance_semi)


    '''
    #compute decision function and compute gradients and store
    X_basic_test_tensor=torch.tensor(X_basic_test, device=device, dtype=torch.float32, requires_grad=True)
    output_basic = torch.tensor(model_basic.decision_function(X_basic_test), device=device, dtype=torch.float32, requires_grad=True)
    y_test_tensor = torch.tensor(y_test, device=device)
    loss = torch.nn.functional.hinge_embedding_loss(output_basic, y_test_tensor.view(-1,1))
    loss.backward()
    gradients_basic = X_basic_test_tensor.grad.numpy()
    
    X_basic_test_tensor = torch.tensor(X_basic_test, device=device, dtype=torch.float32, requires_grad=True)
    output_basic = model_basic(X_basic_test_tensor)  # Assumes model_basic is a PyTorch model
    y_test_tensor = torch.tensor(y_test, device=device)
    loss = torch.nn.functional.hinge_embedding_loss(output_basic, y_test_tensor.view(-1,1))
    loss.backward()
    gradients_basic = X_basic_test_tensor.grad.numpy()

    
    feature_sensitivity_basic.append(gradients_basic.mean(axis=0))
    '''
    '''
    

    # calculate gradients for semi model
    X_semi_test_tensor=torch.tensor(X_semi_test, device=device, dtype=torch.float32, requires_grad=True)
    output_semi = torch.tensor(model_semi.decision_function(X_semi_test), device=device, dtype=torch.float32, requires_grad=True)
    output_semi.backward(torch.ones_like(output_semi))
    gradients_semi = X_semi_test_tensor.grad
    feature_sensitivity_semi.append(gradients_semi.mean(axis=0).numpy())
    '''
pred_basic=pred_basic[1:]
pred_semi=pred_semi[1:]
real_label=real_label[1:]

num_data = np.array(num_data)

score_basic = np.array(score_basic)
score_b=np.sum(num_data*score_basic)

score_semi=np.array(score_semi)
score_s=np.sum(num_data*score_basic)

print("score for basic" + str(score_b))
print("score for semi" + str(score_s))

feature_sensitivity_basic = np.vstack(feature_sensitivity_basic)
mean_importance_basic = np.mean(feature_sensitivity_basic, axis=0)
abs_mean_importance_basic = np.mean(abs(feature_sensitivity_basic), axis=0)

idx_max_basic=np.argsort(mean_importance_basic)[::-1][:5]
idx_min_basic=np.argsort(mean_importance_basic)[::-1][-5:]
idx_absmax_basic=np.argsort(abs_mean_importance_basic)[::-1][:10]

print("basic")
print("max impact " + str(idx_max_basic))
print("max negative impact " + str(idx_min_basic))
print("max abs impact " + str(idx_absmax_basic))

feature_sensitivity_semi = np.vstack(feature_sensitivity_semi)
mean_importance_semi = np.mean(feature_sensitivity_semi, axis=0)
abs_mean_importance_semi = np.mean(abs(feature_sensitivity_semi), axis=0)

idx_max_semi=np.argsort(mean_importance_semi)[::-1][:5]
idx_min_semi=np.argsort(mean_importance_semi)[::-1][-5:]
idx_absmax_semi=np.argsort(abs_mean_importance_semi)[::-1][:10]

print("semi")
print("max impact " + str(idx_max_semi))
print("max negative impact " + str(idx_min_semi))
print("max abs impact " + str(idx_absmax_semi))

#get contigency table and perform mcnemar
contingency_table = np.zeros(2,2)
for i in range(len(real_label)):
    if pred_basic[i] == real_label[i] and pred_semi[i] == real_label[i]:
        contingency_table[0,0] += 1
    if pred_basic[i] != real_label[i] and pred_semi[i] != real_label[i]:
        contingency_table[1,1] += 1
    if pred_basic[i] == real_label[i] and pred_semi[i] != real_label[i]:
        contingency_table[0,1] += 1
    if pred_basic[i] != real_label[i] and pred_semi[i] == real_label[i]:
        contingency_table[1,0] += 1

print(contingency_table)

test_results = mcnemar(contingency_table, exact=True)

print(test_results)

