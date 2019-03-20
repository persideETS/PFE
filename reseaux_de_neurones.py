import os
import csv
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
from sklearn.utils.multiclass import unique_labels
import time

device = torch.device('cpu')
def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = 100.0 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print('Matrice de confusion')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Sortie attendue',
           xlabel='Sortie prédite')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
np.set_printoptions(precision=2)

repertoire = 'C:/Users/gpers_000/Documents/Projet maitrise/Data/Archive CMM Data_Alain/'
colonnes = ['Feat. Type','Feat. Name', 'Value', 'Actual', 'Nominal', 'Dev.', 'Tol-', 'Tol+', 'Out of Tol.', 'Comment']

liste_fichiers = os.listdir(repertoire)
all_data = []

print('Extraction des données ...')
for fichier in liste_fichiers :
    if (os.path.isfile(repertoire + fichier) and (fichier.endswith('.csv') or fichier.endswith('.CSV'))) :
        # lecture du fichier et extraction complète de la matrice
        df = pd.read_csv(repertoire + fichier,
                         names=colonnes,
                         usecols=colonnes,
                         skiprows=1,
                         skipinitialspace=True,
                         encoding = 'windows-1252', quoting=csv.QUOTE_NONNUMERIC)
        # suppression des lignes où les mesures ne sont associées à aucune tolérance ou aucune mesure n'a été faite
        df.dropna(subset=['Actual','Tol-', 'Tol+'], inplace=True)
        if(not df.empty) :
            #['Actual', 'Nominal', 'Dev.', 'Tol-', 'Tol+', 'Out of Tol.']
            all_data.append(df[['Actual', 'Nominal', 'Tol-', 'Tol+', 'Out of Tol.']].astype('float64').fillna(0.0))

print("Fin Extraction données")

data = pd.concat(all_data)

normalized_data = preprocessing.normalize(data[['Actual', 'Nominal', 'Tol-', 'Tol+']], norm='l2')


#### target est la colonne des sorties créée. -1 ou 1 en fonction du contenu de la colonne Out of Tol. 
#### Cela peut être 0 ou 1 aussi. Il faudra sûrement changer la fonction d'activation de la couche de sortie 
data['target'] = -1.0
data.loc[data['Out of Tol.'] != 0.0 , 'target'] = 1.0
labels = data.pop('target')

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels, random_state=0)

print("X_train.shape = ", X_train.shape)
print("X_test.shape = ", X_test.shape)
n_positives = labels[labels==1].size
n_negatives = labels[labels==-1].size
print("classe = 1 : ", n_positives)
print("classe = 0 :  ", n_negatives)

x = torch.tensor(X_train, dtype=torch.float, device=device)
y = torch.tensor(y_train.values, dtype=torch.float, device=device)
y = torch.reshape(y,(y.size()[0],1))

## configuration des réseaux de neurones à modifier 
##model = nn.Sequential(nn.Linear(4, 6),
##                      nn.Linear(6, 1),
##                      nn.Softsign().to(device)) # Tanh() comme fonction d'activation
##
##learning_rate = 3e-3

model = nn.Sequential(nn.Linear(4, 32),
                      nn.Linear(32, 64),
                      nn.Linear(64, 128),
                      nn.Linear(128, 256),
                      nn.Linear(256, 1),
                      nn.Softsign().to(device)) # Tanh() comme fonction d'activation

lr = 0.003
print("lr : ", lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)#Adam
loss_fn = torch.nn.BCEWithLogitsLoss()
all_losses = []
start_time = time.time()

for t in range(100):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print('epoch: ', t+1,' loss: ', loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    all_losses.append(loss) 

print("Temps d'exécution : %s seconds ---" % (time.time() - start_time))

plt.figure()  
plt.plot(all_losses)
plt.show()

xPredicted = torch.tensor(X_test, dtype=torch.float)
yToPredict = torch.tensor(y_test.values, dtype=torch.float)
yToPredict = torch.reshape(yToPredict,(yToPredict.size()[0],1))
yPredicted = model(xPredicted)

results = np.concatenate((yToPredict.cpu().data.numpy(), yPredicted.cpu().data.numpy()), axis=1)
np.savetxt('C:/Users/gpers_000/Documents/Projet maitrise/results_validation_softsign.csv', results, delimiter=',')

print("Données 2019")
fichier = 'C:/Users/gpers_000/Documents/Projet maitrise/Data/BaseCmm_Data_Report_2019.csv'
# lecture du fichier et extraction complète de la matrice
df = pd.read_csv(fichier, decimal=',', encoding = 'windows-1252')

# suppression des lignes où les mesures ne sont associées à aucune tolérance ou aucune mesure n'a été faite
df.drop(df[df['TolerancePositive']==0.0].index, inplace=True)
df.drop(df[df['ValeurActuelle']==0.0].index, inplace=True)
if(not df.empty) :
    df['target']= -1.0
    df.loc[df['HorsTolerance'] != 0.0 , 'target'] = 1.0
    y = df.pop('target')
    X_normalized = preprocessing.normalize(df[['ValeurActuelle', 'ValeurNominale', 'ToleranceNegative', 'TolerancePositive']], norm='l2')#normalized_data[['ValeurNominale', 'ToleranceNegative', 'TolerancePositive']]
    xPredicted = torch.tensor(X_normalized, dtype=torch.float)
    yToPredict = torch.tensor(y.values, dtype=torch.float)
    yToPredict = torch.reshape(yToPredict,(yToPredict.size()[0],1))
    y_pred_rf_2019 = model(xPredicted)

    yToPredict = yToPredict.cpu().data.numpy()
    y_pred_rf_2019 = np.rint(y_pred_rf_2019.cpu().data.numpy())
    results = np.concatenate((yToPredict, y_pred_rf_2019), axis=1)
    np.savetxt('C:/Users/gpers_000/Documents/Projet maitrise/results_tests_softsign.csv', results, delimiter=',')    
    score_rf_2019 = precision_recall_fscore_support(yToPredict, y_pred_rf_2019, average='weighted')
    accuracy_2019 = accuracy_score(yToPredict, y_pred_rf_2019, normalize =True) 
    print("scores : \n", score_rf_2019)
    print("accuracy score : \n", accuracy_2019)
    plot_confusion_matrix(yToPredict, y_pred_rf_2019, title='Matrice de confusion')
    plt.show()