import os
import csv
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import pydot

#### Matrice de confusion
def plot_confusion_matrix(y_true, y_pred, classes,
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
    print(unique_labels(y_true, y_pred))
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

np.set_printoptions(precision=4)

##### À modifier en fonction de l'emplacement de vos base d'apprentissage et de validation
repertoire = 'C:/Users/gpers_000/Documents/Projet maitrise/Data/Archive CMM Data_Alain/'

colonnes = ['Feat. Type','Feat. Name', 'Value', 'Actual', 'Nominal', 'Dev.', 'Tol-', 'Tol+', 'Out of Tol.', 'Comment']

liste_fichiers = os.listdir(repertoire)

all_data = []

print('Extraction données ...')
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

#### normalisation
normalized_data = preprocessing.normalize(data[['Actual', 'Nominal', 'Tol-', 'Tol+']], norm='l2')

data['target'] = -1
data.loc[data['Out of Tol.'] != 0.0 , 'target'] = 1.0
labels = data.pop('target')

# séparation en base d'apprentissage et de validation
X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels,  test_size = 0.3, random_state = 0)

print("X_train.shape = ", X_train.shape)
print("X_test.shape = ", X_test.shape)
print("classe = 1 train : ", y_train[y_train==1].size)
print("classe = -1 train : ", y_train[y_train==-1].size)
print("classe = 1 test : ", y_test[y_test==1].size)
print("classe = -1 test :  ", y_test[y_test==-1].size)
print("classe = 1 : ", labels[labels==1].size)
print("classe = -1 :  ", labels[labels==-1].size)

print("Random forests")
#Variation du nombre d'arbres
arbres = [1, 5, 25, 75, 100]
for nb_arbres in arbres :
    print("Nombre d'arbres : ", nb_arbres)
    classifier_rf = RandomForestClassifier(n_estimators=nb_arbres) 
    start_time = time.time()
    #print("cross validation score : \n", cross_val_score(classifier_rf, X_train, y_train, cv=10))
    classifier_rf.fit(X_train, y_train)
    print("Temps d'exécution : %s seconds ---" % (time.time() - start_time))
    y_pred_rf = classifier_rf.predict(X_test)
    scores_rf = precision_recall_fscore_support(y_test, y_pred_rf, average='weighted')
    print("scores : \n", scores_rf)    
    print("Fin RF ...\n")

    ##### Pour afficher le 1er arbre de la forêt
    ##### À modifier en fonction de l'emplacement de vos base de données
    ##### utiliser ces 2 lignes si vous voulez voir un arbre de décision de votre forêt 
    ##### nécessite la librairie pydot https://pypi.org/project/pydot/ 
    tree = classifier_rf.estimators_[0]
    export_graphviz(tree, out_file = 'C:/Users/gpers_000/Documents/Projet maitrise/tree_4_features_'+str(nb_arbres)+'_.dot', feature_names = ['Actual', 'Nominal', 'Tol-', 'Tol+'], rounded = True, precision = 1)

    # importance des caractéristiques
    importances = list(classifier_rf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(['Actual', 'Nominal', 'Tol-', 'Tol+'], importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

    # colonnes = ['NomPiece', 'NomProgramme', 'NomVariable', 'TypeVariable', 'NomCaracteristique', 'Date', 'DateHeure', 'ValeurActuelle', 'ValeurNominale', 'ValeurDeviation', 'ToleranceNegative', 'TolerancePositive', 'HorsTolerance', 'LSL', 'USL', 'LCL', 'UCL', 'CP', 'CPU', 'CPL', 'CPK', 'Etendu', 'Evenement', 'Commentaire', 'AutreCommentaire', 'Valid']
    print("Données 2019")
    ##### À modifier en fonction de l'emplacement de vos base de tests
    fichier = 'C:/Users/gpers_000/Documents/Projet maitrise/Data/BaseCmm_Data_Report_2019.csv'
    # lecture du fichier et extraction complète de la matrice
    df = pd.read_csv(fichier, decimal=',', encoding = 'windows-1252')

    # suppression des lignes où les mesures ne sont associées à aucune tolérance ou aucune mesure n'a été faite
    # c'est un peu différent de la manipulation faite pour la base de tests
    df.drop(df[df['TolerancePositive']==0.0].index, inplace=True)
    df.drop(df[df['ValeurActuelle']==0.0].index, inplace=True)
    if(not df.empty) :
        #['Actual', 'Nominal', 'Dev.', 'Tol-', 'Tol+', 'Out of Tol.']
        df['target']= -1
        df.loc[df['HorsTolerance'] != 0.0 , 'target'] =  1.0
        y = df.pop('target')
        X_normalized = preprocessing.normalize(df[['ValeurActuelle', 'ValeurNominale', 'ToleranceNegative', 'TolerancePositive']], norm='l2')#normalized_data[['ValeurNominale', 'ToleranceNegative', 'TolerancePositive']]
        
        y_pred_rf_2019 = classifier_rf.predict(X_normalized)
        score_rf_2019 = precision_recall_fscore_support(y, y_pred_rf_2019, average='weighted')
        accuracy_2019 = accuracy_score(y, y_pred_rf_2019, normalize =True) 
        print("scores : \n", score_rf_2019)
        print("accuracy score : \n", accuracy_2019)
        plot_confusion_matrix(y, y_pred_rf_2019, classes=['Conforme', 'Non-Conforme'], title='Matrice de confusion, Nombre arbres = '+str(nb_arbres))
        plt.show()