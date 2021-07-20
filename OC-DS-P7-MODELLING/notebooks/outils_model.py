# -*- coding: utf-8 -*-
""" Librairie personnelle pour manipulation les modèles de machine learning
"""

# ====================================================================
# Outils ML -  projet 4 Openclassrooms
# Version : 0.0.0 - CRE LR 23/03/2021
# ====================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import time
import pickle
import shap
import outils_data
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, \
    explained_variance_score, median_absolute_error
from sklearn.model_selection import cross_validate, RandomizedSearchCV, \
    GridSearchCV, learning_curve  # , cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, \
    BayesianRidge, HuberRegressor, \
    OrthogonalMatchingPursuit, Lars, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, \
    ExtraTreesRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import eli5
from eli5.sklearn import PermutationImportance
from pprint import pprint


# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.0'


# --------------------------------------------------------------------
# -- Entrainer/predire modele de regression de base avec cross-validation
# --------------------------------------------------------------------


def process_regression(
        model_reg,
        X_train,
        X_test,
        y_train,
        y_test,
        df_resultats,
        titre,
        affiche_tableau=True,
        affiche_comp=True,
        affiche_erreur=True,
        xlim_sup=130000000):
    """
    Lance un modele de régression, effectue cross-validation et sauvegarde les
    performances
    Parameters
    ----------
    model_reg : modèle de régression initialisé, obligatoire.
    X_train : train set matrice X, obligatoire.
    X_test : test set matrice X, obligatoire.
    y_train : train set vecteur y, obligatoire.
    y_test : test set, vecteur y, obligatoire.
    df_resultats : dataframe sauvegardant les traces, obligatoire
    titre : titre à inscrire dans le tableau de sauvegarde, obligatoire.
    affiche_tableau : booleen affiche le tableau de résultat, facultatif.
    affiche_comp : booleen affiche le graphique comparant y_test/y_pres,
                   facultatif.
    affiche_erreur : booleen affiche le graphique des erreurs, facultatif.
    xlim_sup : limite supérieure de x, facultatif.
    Returns
    -------
    df_resultats : Le dataframe de sauvegarde des performances.
    y_pred : Les prédictions pour le modèle
    """
    # Top début d'exécution
    time_start = time.time()

    # Entraînement du modèle
    model_reg.fit(X_train, y_train)

    # Sauvegarde du modèle de régression entaîné
    with open('modeles/modele_' + titre + '.pickle', 'wb') as f:
        pickle.dump(model_reg, f, pickle.HIGHEST_PROTOCOL)

    # Prédictions avec le test set
    y_pred = model_reg.predict(X_test)

    # Top fin d'exécution
    time_end = time.time()

    # Pour que le R2 soit représentatif des valeurs réelles
    y_test_nt = (10 ** y_test) + 1
    y_pred_nt = (10 ** y_pred) + 1

    # Calcul des métriques
    mae = mean_absolute_error(y_test_nt, y_pred_nt)
    mse = mean_squared_error(y_test_nt, y_pred_nt)
    rmse = sqrt(mse)
    r2 = r2_score(y_test_nt, y_pred_nt)
    errors = abs(y_pred - y_test_nt)
    mape = 100 * np.mean(errors / y_test_nt)
    accuracy = 100 - mape

    # durée d'exécution
    time_execution = time_end - time_start

    # cross validation
    scoring = ['r2', 'neg_mean_squared_error']
    scores = cross_validate(model_reg, X_train, y_train, cv=10,
                            scoring=scoring, return_train_score=True)

    # Sauvegarde des performances
    df_resultats = df_resultats.append(pd.DataFrame({
        'Modèle': [titre],
        'R2': [r2],
        'MSE': [mse],
        'RMSE': [rmse],
        'MAE': [mae],
        'Erreur moy': [np.mean(errors)],
        'Précision': [accuracy],
        'Durée': [time_execution],
        'Test R2 CV': [scores['test_r2'].mean()],
        'Test R2 +/-': [scores['test_r2'].std()],
        'Test MSE CV': [-(scores['test_neg_mean_squared_error'].mean())],
        'Train R2 CV': [scores['train_r2'].mean()],
        'Train R2 +/-': [scores['train_r2'].std()],
        'Train MSE CV': [-(scores['train_neg_mean_squared_error'].mean())]
    }), ignore_index=True)

    if affiche_tableau:
        display(df_resultats.style.hide_index())

    if affiche_comp:
        # retour aux valeurs d'origine
        test = (10 ** y_test) + 1
        predict = (10 ** y_pred) + 1

        # Affichage Test vs Predictions
        sns.jointplot(
            test,
            predict,
            kind='reg')
        plt.xlabel('y_test')
        plt.ylabel('y_predicted')
        plt.suptitle(t='Tests /Predictions pour : '
                       + str(titre),
                       y=0,
                       fontsize=16,
                       alpha=0.75,
                       weight='bold',
                       ha='center')
        plt.xlim([0, xlim_sup])
        plt.show()

    if affiche_erreur:
        # retour aux valeurs d'origine
        test = (10 ** y_test) + 1
        predict = (10 ** y_pred) + 1
        # affichage des erreurs
        df_res = pd.DataFrame({'true': test, 'pred': predict})
        df_res = df_res.sort_values('true')

        plt.plot(df_res['pred'].values, label='pred')
        plt.plot(df_res['true'].values, label='true')
        plt.xlabel('Test set')
        plt.ylabel("Consommation energie totale")
        plt.suptitle(t='Erreurs pour : '
                     + str(titre),
                     y=0,
                     fontsize=16,
                     alpha=0.75,
                     weight='bold',
                     ha='center')
        plt.legend()
        plt.show()

    return df_resultats, y_pred

# --------------------------------------------------------------------
# -- Modèles de régression - entraîner le modèle de base et scores
# --------------------------------------------------------------------


def comparer_baseline_regressors(
        X,
        y,
        cv=10,
        metrics=[
            'r2',
            'neg_mean_squared_error'],
        seed=21):
    """Comparaison rapide des modèles de régression de base.
    Parameters
    ----------
    X: Matrice x, obligatoire
    y: Target vecteur, obligatoire
    cv: le nombre de k-folds pour la cross validation, optionnel
    metrics: liste des scores à appliquer, optionnel
    seed: nombre aléatoire pour garantir la reproductibilité des données.
    Returns
    -------
    La liste des modèles avec les scores
    """
    # Les listes des modèles de régression de base (à enrichir)
    models = []
    models.append(('dum_mean', DummyRegressor(strategy='mean')))
    models.append(('dum_med', DummyRegressor(strategy='median')))
    models.append(('lin', LinearRegression()))
    models.append(
        ('ridge',
         Ridge(
             alpha=10,
             solver='cholesky',
             random_state=seed)))
    models.append(('lasso', Lasso(random_state=seed)))
    models.append(('en', ElasticNet(random_state=seed)))
    models.append(('svr', SVR()))
    models.append(('br', BayesianRidge()))
    models.append(('hr', HuberRegressor()))
    models.append(('omp', OrthogonalMatchingPursuit()))
    models.append(('lars', Lars(random_state=seed)))
    models.append(('knr', KNeighborsRegressor()))
    models.append(('dt', DecisionTreeRegressor(random_state=seed)))
    models.append(('ada', AdaBoostRegressor(random_state=seed)))
    models.append(('xgb', XGBRegressor(seed=seed)))
    models.append(('sgd', SGDRegressor(random_state=seed)))
    models.append(('lgbm', LGBMRegressor(random_state=seed)))
    models.append(('rfr', RandomForestRegressor(random_state=seed)))
    models.append(('etr', ExtraTreesRegressor(random_state=seed)))
    models.append(('cat', CatBoostRegressor(random_state=seed, verbose=False)))
    models.append(('gbr', GradientBoostingRegressor(random_state=seed)))
    models.append(('bag', BaggingRegressor(random_state=seed)))

    # Création d'un dataframe stockant les résultats des différents algorithmes
    df_resultats = pd.DataFrame(dtype='object')
    for name, model in models:

        # Cross validation d'entraînement du modèle
        scores = pd.DataFrame(
            cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=metrics,
                return_train_score=True))

        # Sauvegarde des performances
        df_resultats = df_resultats.append(pd.DataFrame({
            'Modèle': [name],
            'Fit time': [scores['fit_time'].mean()],
            'Durée': [scores['score_time'].mean()],
            'Test R2 CV': [scores['test_r2'].mean()],
            'Test R2 +/-': [scores['test_r2'].std()],
            'Test MSE CV': [-(scores['test_neg_mean_squared_error'].mean())],
            'Train R2 CV': [scores['train_r2'].mean()],
            'Train R2 +/-': [scores['train_r2'].std()],
            'Train MSE CV': [-(scores['train_neg_mean_squared_error'].mean())]
        }), ignore_index=True)
        print(f'Exécution terminée - Modèle : {name}')

    return df_resultats.sort_values(
        by=['Test R2 CV', 'Test MSE CV', 'Durée'], ascending=False)

# --------------------------------------------------------------------
# COMMUN A TOUS LES MODELES : évaluation de l'hyperparamètre
# --------------------------------------------------------------------


def evaluer_hyperparametre(models, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre n_estimators de ExraTreesRegressor
    Parameters
    ----------
    models : liste des modèles instanciés avec des valeurs différentes
             d'hyperparamètre', obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    # sauvegarde des performances
    results, names = list(), list()

    print('Hyperparam', 'Test R2 +/- std', 'Train R2 +/- std')
    for name, model in models.items():
        # evaluate the model
        # scores = cross_val_score(model, X, y, scoring='r2', cv=10, n_jobs=-1)
        scores = pd.DataFrame(
            cross_validate(
                model,
                X,
                y,
                cv=10,
                scoring='r2',
                return_train_score=True))

        # store the results
        results.append(scores['test_score'])
        names.append(name)
        test_mean = scores['test_score'].mean()
        test_std = scores['test_score'].std()
        train_mean = scores['train_score'].mean()
        train_std = scores['train_score'].std()
        # sAffiche le R2 pour le nombre d'arbres
        print('>%s %.5f (%.5f) %.5f (%.5f)' %
              (name, test_mean, test_std, train_mean, train_std))

    if affiche_boxplot:
        plt.boxplot(results, labels=names, showmeans=True)
        plt.show()

# --------------------------------------------------------------------
# ET - ExtaTreesRegressor - règle l'hyper-paramètre max_features
# nombre d'arbres
# --------------------------------------------------------------------


def regle_extratrees_nestimators(n_estimators, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre n_estimators de ExraTreesRegressor
    Parameters
    ----------
    n_estimators : nombre d'arbres, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    for n in n_estimators:
        models[str(n)] = ExtraTreesRegressor(n_estimators=n, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre max_features
# NOMBRE DE CARACTERISTIQUES
# --------------------------------------------------------------------


def regle_extratrees_maxfeatures(max_features, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre max_features de ExraTreesRegressor
    Parameters
    ----------
    max_features : NOMBRE DE CARACTERISTIQUES, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_features:
        models[str(i)] = ExtraTreesRegressor(max_features=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre min_samples_split
# nombre minimum d'échantillons requis pour diviser un nœud interne
# --------------------------------------------------------------------

def regle_extratrees_minsamplessplit(
        min_samples_split, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre min_samples_split de ExraTreesRegressor
    Parameters
    ----------
    min_samples_split : nombre minimum d'échantillons requis pour diviser un
    nœud interne, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in min_samples_split:
        models[str(i)] = ExtraTreesRegressor(
            min_samples_split=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre max_depth
# profondeur maximale de l'arbre
# --------------------------------------------------------------------

def regle_extratrees_maxdepth(max_depth, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre max_depth de ExraTreesRegressor
    Parameters
    ----------
    max_depth : profondeur maximale de l'arbre, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_depth:
        models[i] = ExtraTreesRegressor(max_depth=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre criterion
# mesurer la qualité d'un fractionnement
# --------------------------------------------------------------------

def regle_extratrees_criterion(criterion, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre criterion de ExraTreesRegressor
    Parameters
    ----------
    criterion : mesurer la qualité d'un fractionnement, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for s in criterion:
        models[s] = ExtraTreesRegressor(criterion=s, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre min_samples_leaf
# nombre minimum d'échantillons requis pour se trouver à un nœud de feuille
# --------------------------------------------------------------------

def regle_extratrees_minsamplesleaf(
        min_samples_leaf, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre min_samples_leaf de ExraTreesRegressor
    Parameters
    ----------
    min_samples_leaf : nombre minimum d'échantillons requis pour se trouver à
                       un nœud de feuille, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in min_samples_leaf:
        models[i] = ExtraTreesRegressor(min_samples_leaf=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre max_leaf_nodes
# Un nœud sera divisé si cette division induit une diminution de l'impureté
# supérieure ou égale à cette valeur
# --------------------------------------------------------------------

def regle_extratrees_maxleafnodes(max_leaf_nodes, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre max_leaf_nodes de ExraTreesRegressor
    Parameters
    ----------
    max_leaf_nodes : Un nœud sera divisé si cette division induit une
    diminution de l'impureté supérieure ou égale à cette valeur.
    Faire croître les arbres avec max_leaf_nodes de la manière la plus
    efficace possible. Les meilleurs nœuds sont définis comme une réduction
    relative de l'impureté. Si None, le nombre de nœuds feuilles est illimité,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_leaf_nodes:
        models[i] = ExtraTreesRegressor(max_leaf_nodes=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre min_impurity_decrease
# Un nœud sera divisé si cette division induit une diminution de l'impureté
# supérieure ou égale à cette valeur
# --------------------------------------------------------------------


def regle_extratrees_minimpuritydecrease(
        min_impurity_decrease, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre min_impurity_decrease de ExraTreesRegressor
    Parameters
    ----------
    min_impurity_decrease : Un nœud sera divisé si cette division induit une
    diminution de l'impureté plus grande ou égale à cette valeur., obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in min_impurity_decrease:
        models[i] = ExtraTreesRegressor(
            min_impurity_decrease=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre bootstrap
# Si les échantillons bootstrap sont utilisés lors de la construction
# des arbres. Si Faux, l'ensemble des données est utilisé pour construire chaque arbre.
# --------------------------------------------------------------------


def regle_extratrees_bootstrap(bootstrap, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre bootstrap de ExraTreesRegressor
    Parameters
    ----------
    bootstrap : Si les échantillons bootstrap sont utilisés lors de la construction des arbres.
    Si Faux, l'ensemble des données est utilisé pour construire chaque arbre,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for b in bootstrap:
        models[b] = ExtraTreesRegressor(bootstrap=b, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre warm_start
# Lorsqu'elle est définie sur True, la solution de l'appel précédent à
# l'ajustement est réutilisée et d'autres estimateurs sont ajoutés à l'ensemble,
# sinon, une nouvelle forêt est ajustée.
# --------------------------------------------------------------------


def regle_extratrees_warm_start(warm_start, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre warm_start de ExraTreesRegressor
    Parameters
    ----------
    warm_start : Lorsqu'elle est définie sur True, la solution de l'appel
    précédent à l'ajustement est réutilisée et d'autres estimateurs sont
    ajoutés à l'ensemble, sinon, une nouvelle forêt est ajustée., obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for b in warm_start:
        models[b] = ExtraTreesRegressor(warm_start=b, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre max_samples
# Si bootstrap est True, le nombre d'échantillons à tirer de X pour entraîner
# chaque estimateur de base
# --------------------------------------------------------------------


def regle_extratrees_maxsamples(max_samples, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre warm_start de ExraTreesRegressor
    Parameters
    ----------
    max_samples :  Si bootstrap est True, le nombre d'échantillons à tirer
    de X pour entraîner chaque estimateur de base, obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_samples:
        models[i] = ExtraTreesRegressor(max_samples=i, bootstrap=True,
                                        random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - règle l'hyper-paramètre ccp_alpha
# Paramètre de complexité utilisé pour l'élagage minimal de complexité-coût.
# Le sous-arbre avec la plus grande complexité de coût qui est plus petite que
 # ccp_alpha sera choisi. Par défaut, aucun élagage n'est effectué
# --------------------------------------------------------------------

def regle_extratrees_ccpalpha(ccp_alpha, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre ccp_alpha de ExraTreesRegressor
    Parameters
    ----------
    ccp_alpha : Paramètre de complexité utilisé pour l'élagage minimal de
    complexité-coût. Le sous-arbre avec la plus grande complexité de coût qui
    est plus petite que ccp_alpha sera choisi. Par défaut, aucun élagage n'est
    effectué, obligatoire.
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in ccp_alpha:
        models[i] = ExtraTreesRegressor(ccp_alpha=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - RANDOMIZED SEARCH CV
# --------------------------------------------------------------------


def extratreesregressor_randomized_search_cv(x_train, y_train, x_test, y_test,
                                             seed=21):
    '''
    Randomized Search Cv pour le modèle ExtraTreesRegressor
    Parameters
    ----------
    x_train : jeu d'entraînement matrice X, obligatoire
    y_train : jeu d'entraînement target y, obligatoire
    x_test : jeu de test matrice X, obligatoire.
    y_test : jeu de test target y, obligatoire.
    seed : nombre aléatoire pour la reproductibilité des résultats, facultatif
    Returns
    -------
    best_et_random : Le modèl trouvé aléatoirement avec RandomizedSearchCV
    '''

    # Définition  de la grille
    # ----------------------------------------------------------------------
    # Nombre d'arbres
    n_estimators = [int(x) for x in np.linspace(start=260, stop=310, num=6)]
    # Nombre de variables à considérer à chaque split de décision
    max_features = ['auto']
    max_features.append(None)
    # Nombre maximum de niveau de l'arbre
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Nombre minimum d'échantillons requis pour chaque split de décision
    min_samples_split = [1, 2, 3, 4, 5, 6, 10, 15, 20]
    # Nombre minimum d'échantillons requis pour chaque feuille
    min_samples_leaf = [1, 2, 3, 4, 5, 6, 10, 15]
    # Grille de recherche
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    print('Grille de recherche : \n')
    pprint(random_grid)

    # Instanciation du modèle
    et_rand = ExtraTreesRegressor()
    # Recherche aléatoires de paramètres avec 3 folds de cross validation
    # pour 100 itérations avec différentes combinaisons utilisant tous les
    # processeurs
    et_random = RandomizedSearchCV(estimator=et_rand,
                                   param_distributions=random_grid,
                                   n_iter=100, cv=3, verbose=0,
                                   random_state=seed, n_jobs=-1)
    # Entraînement du modèle
    et_random.fit(x_train, y_train)

    # Affiche les meilleurs hyperparamètres
    print('\nMeilleurs hyperparamètres : \n')
    pprint(et_random.best_params_)

    # Evaluation du modèle issu de RandomizedSearchCV
    best_et_random = et_random.best_estimator_
    evaluate(best_et_random, x_test, y_test)

    # Calcul de l'importance de chaque variables
    feature_et_importance = best_et_random.feature_importances_
    sorted_idx = feature_et_importance.argsort()
    feature_et_importance_tri = outils_data.sort_array(feature_et_importance)
    # Visualisation des features importance
    plot_features_importance(feature_et_importance_tri,
                             x_train.columns[sorted_idx])

    return best_et_random

# --------------------------------------------------------------------
# ET- EXTRA-TREE REGRESSOR - GRID SEARCH CV
# --------------------------------------------------------------------


def extratreesregressor_grid_search_cv(x_train, y_train, x_test, y_test):
    '''
    Grid Search Cv pour le modèle ExtraTreesRegressor
    Parameters
    ----------
    x_train : jeu d'entraînement matrice X, obligatoire
    y_train : jeu d'entraînement target y, obligatoire
    x_test : jeu de test matrice X, obligatoire.
    y_test : jeu de test target y, obligatoire.
    Returns
    -------
    best_et_grid : Le modèl trouvé aléatoirement avec GridSearchCV
    '''
    # Grille de paramètre
    param_grid = {'n_estimators': [270, 280, 300],
                  'min_samples_leaf': [2, 4],
                  'max_features': ['auto', 'None'],
                  'min_samples_split': [10, 15, 20],
                  'max_depth': [50, 100, 'None']
                  }
    print('Grille de recherche : \n')
    pprint(param_grid)

    # Instanciation du modèle
    etr_grid = ExtraTreesRegressor()

    # Instanciation de la recherche sur grille avec validation croisée
    grid_search = GridSearchCV(
        estimator=etr_grid, param_grid=param_grid, cv=10, n_jobs=-1, verbose=0)

    # Entraînement de la GridSearchCV
    grid_result_et = grid_search.fit(x_train, y_train.values.ravel())

    # Sauvegarde du modèle
    with open('modeles/modele_extrtreesregressor_grid_man.pickle', 'wb') as f:
        pickle.dump(grid_result_et, f, pickle.HIGHEST_PROTOCOL)

    print(
        f'\nBest score : {grid_result_et.best_score_} \n\navec les hyperparamètres :\n{grid_result_et.best_params_}')

    best_et_grid = ExtraTreesRegressor(**grid_result_et.best_params_)

    # Entraînement du modèle
    best_et_grid.fit(x_train, y_train)

    evaluate(best_et_grid, x_test, y_test)

    return best_et_grid

# --------------------------------------------------------------------
# CAT- CATBOOST REGRESSOR - règle l'hyper-paramètre iterations
# Le nombre maximum d'arbres à construire, la valeur par défaut est 1000.
# --------------------------------------------------------------------


def regle_catboost_iterations(iterations, categorical_features_index,
                              X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre iterations de CatBoostRegressor
    Parameters
    ----------
    iterations :Le nombre maximum d'arbres à construire, la valeur par défaut
    est 1000.,
    obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in iterations:
        models[i] = CatBoostRegressor(cat_features=categorical_features_index,
                                      iterations=i, random_state=21, verbose=0)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# CAT- CATBOOST REGRESSOR - règle l'hyper-paramètre learning_rate
# Le taux d'apprentissage qui détermine la rapidité ou la lenteur de
# l'apprentissage du modèle
# --------------------------------------------------------------------


def regle_catboost_learningrate(learning_rate, categorical_features_index,
                                X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre ccp_alpha de ExraTreesRegressor
    Parameters
    ----------
    learning_rate : Le taux d'apprentissage qui détermine la rapidité ou la
    lenteur de l'apprentissage du modèle, obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in learning_rate:
        models[i] = CatBoostRegressor(
            cat_features=categorical_features_index,
            learning_rate=i,
            random_state=21,
            verbose=0)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# CAT- CATBOOST REGRESSOR - règle l'hyper-paramètre depth
# La profondeur de l'arbre
# --------------------------------------------------------------------


def regle_catboost_depth(depth, categorical_features_index,
                         X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre ccp_alpha de ExraTreesRegressor
    Parameters
    ----------
    depth : La profondeur de l'arbre, obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in depth:
        models[i] = CatBoostRegressor(depth=i,
                                      cat_features=categorical_features_index,
                                      random_state=21, verbose=0)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# CAT- CATBOOST REGRESSOR - règle l'hyper-paramètre loss_function
# Métrique utilisée pour la formation.
# --------------------------------------------------------------------


def regle_catboost_lossfunction(loss_function, categorical_features_index,
                                X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre ccp_alpha de ExraTreesRegressor
    Parameters
    ----------
    loss_function : Métrique utilisée pour la formation., obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for s in loss_function:
        models[s] = CatBoostRegressor(loss_function=s,
                                      cat_features=categorical_features_index,
                                      random_state=21, verbose=0)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# CAT- CATBOOST REGRESSOR - règle l'hyper-paramètre l2_leaf_reg
# Coefficient du terme de régularisation L2 de la fonction de coût
# --------------------------------------------------------------------


def regle_catboost_l2leafreg(l2_leaf_reg, categorical_features_index,
                             X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre ccp_alpha de ExraTreesRegressor
    Parameters
    ----------
    l2_leaf_reg : Coefficient du terme de régularisation L2 de la fonction de
    coût, obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in l2_leaf_reg:
        models[i] = CatBoostRegressor(l2_leaf_reg=i,
                                      cat_features=categorical_features_index,
                                      random_state=21, verbose=0)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# CAT- CAT BOOST REGRESSOR - RANDOMIZED SEARCH CV
# --------------------------------------------------------------------

def catboostregressor_randomized_search_cv(
        x_train,
        y_train,
        x_test,
        y_test,
        categorical_features_index,
        seed=21):
    '''
    Randomized Search Cv pour le modèle ExtraTreesRegressor
    Parameters
    ----------
    x_train : jeu d'entraînement matrice X, obligatoire
    y_train : jeu d'entraînement target y, obligatoire
    x_test : jeu de test matrice X, obligatoire.
    y_test : jeu de test target y, obligatoire.
    categorical_features_index : les variables qualitatives
    seed : nombre aléatoire pour la reproductibilité des résultats, facultatif
    Returns
    -------
    best_cat_random : Le modèl trouvé aléatoirement avec GridSearchCV
    '''

    # Définition  de la grille
    # ----------------------------------------------------------------------
    # Nombre d'arbres
    iterations = [1000, 1100]
    # Taux d'apprentissage
    learning_rate = [0.03, 0.04]
    # Profondeur de l'arbre
    depth = [4, 5, 6]
    # Métrique utilisée pour la formation
    # loss_function = ['RMSE', 'MAPE']
    # Coefficient du terme de régularisation L2 de la fonction de coût
    l2_leaf_reg = [1, 2, 3]
    # Grille de recherche
    cat_random_grid = {'iterations': iterations,
                       'learning_rate': learning_rate,
                       'depth': depth,
                       # 'loss_function': loss_function,
                       'l2_leaf_reg': l2_leaf_reg}
    print('Grille de recherche : \n')
    pprint(cat_random_grid)

    # Instanciation du modèle
    cat_rand = CatBoostRegressor(cat_features=categorical_features_index)
    # Recherche aléatoires de paramètres avec 3 folds de cross validation
    # pour 100 itérations avec différentes combinaisons utilisant tous les
    # processeurs
    cat_random = RandomizedSearchCV(estimator=cat_rand,
                                    param_distributions=cat_random_grid,
                                    n_iter=100, cv=3, verbose=False,
                                    random_state=seed, n_jobs=-1)
    # Entraînement du modèle
    cat_random.fit(x_train, y_train, verbose=False)

    # Affiche les meilleurs hyperparamètres
    print('\nMeilleurs hyperparamètres : \n')
    pprint(cat_random.best_params_)

    # Evaluation du modèle issu de RandomizedSearchCV
    best_cat_random = cat_random.best_estimator_
    evaluate(best_cat_random, x_test, y_test)

    # Calcul de l'importance de chaque variables
    df_feature_cat_importance = CatBoostRegressor.get_feature_importance(
        best_cat_random, prettified=True)
    df_feature_cat_importance.sort_values(by='Importances', ascending=True,
                                          inplace=True)
    nom_cat_variables = df_feature_cat_importance['Feature Id'].values
    feature_cat_importance = df_feature_cat_importance['Importances'].values
    # Passage en pourcentage
    feature_cat_importance = [val / 100 if val !=
                              0 else 0 for val in feature_cat_importance]
    # Visualisation des features importance
    plot_features_importance(feature_cat_importance, nom_cat_variables)

    return best_cat_random

# --------------------------------------------------------------------
# CAT- CAT BOOST REGRESSOR - GRID SEARCH CV
# --------------------------------------------------------------------


def catboostregressor_grid_search_cv(x_train, y_train, x_test, y_test,
                                     categorical_features_index):
    '''
    Grid Search Cv pour le modèle ExtraTreesRegressor
    Parameters
    ----------
    x_train : jeu d'entraînement matrice X, obligatoire
    y_train : jeu d'entraînement target y, obligatoire
    x_test : jeu de test matrice X, obligatoire.
    y_test : jeu de test target y, obligatoire.
    categorical_features_index : les variables catégorielle
    Returns
    -------
    best_et_grid : Le modèl trouvé aléatoirement avec RandomizedSearchCV
    '''
    # Grille de paramètre
    param_cat_grid = {'learning_rate': [0.03, 0.04],
                      'depth': [5, 6],
                      'l2_leaf_reg': [3, 4]
                      }
    print('Grille de recherche : \n')
    pprint(param_cat_grid)

    # Instanciation du modèle
    cat_grid = CatBoostRegressor(iterations=1100,
                                 cat_features=categorical_features_index,
                                 verbose=False, )

    # Instanciation de la recherche sur grille avec validation croisée
    grid_cat_search = GridSearchCV(
        estimator=cat_grid,
        param_grid=param_cat_grid,
        cv=10,
        n_jobs=-1,
        verbose=False)

    # Entraînement de la GridSearchCV
    grid_result_cat_man = grid_cat_search.fit(x_train, y_train)

    # Sauvegarde du modèle
    with open('modeles/modele_catboostregressor_grid_man.pickle', 'wb') as f:
        pickle.dump(grid_result_cat_man, f, pickle.HIGHEST_PROTOCOL)

    print(
        f'Best score : {grid_result_cat_man.best_score_} \n\navec les hyperparamètres :\n{grid_result_cat_man.best_params_}')

    best_cat_model = CatBoostRegressor(cat_features=categorical_features_index,
                                       verbose=False,
                                       **grid_result_cat_man.best_params_)

    # Entraînement du modèle
    best_cat_model.fit(x_train, y_train)

    evaluate(best_cat_model, x_test, y_test)

    return best_cat_model


# --------------------------------------------------------------------
# GB- GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre n_estimators
# Le nombre d'étapes de boosting à exécuter
# --------------------------------------------------------------------

def regle_gradboost_nestimators(n_estimators, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre ccp_alpha de GradientBoostRegressor
    Parameters
    ----------
    n_estimators : Le nombre d'étapes de boosting à exécuter, obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in n_estimators:
        models[i] = GradientBoostingRegressor(n_estimators=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# GB- GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre learning_rate
#  Le taux d'apprentissage
# --------------------------------------------------------------------


def regle_gradboost_learningrate(learning_rate, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre learning_rate de GradientBoostRegressor
    Parameters
    ----------
    learning_rate :  Le taux d'apprentissage, obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in learning_rate:
        models[i] = GradientBoostingRegressor(learning_rate=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# GB- GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre criterion
# La fonction permettant de mesurer la qualité d'un fractionnement
# --------------------------------------------------------------------


def regle_gradboost_criterion(criterion, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre criterion de GradientBoostRegressor
    Parameters
    ----------
    criterion : La fonction permettant de mesurer la qualité d'un
    fractionnement, obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for s in criterion:
        models[s] = GradientBoostingRegressor(criterion=s, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# GB- GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre max_depth
# Profondeur maximale des estimateurs de régression individuels
# --------------------------------------------------------------------

def regle_gradboost_maxdepth(max_depth, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre criterion de GradientBoostRegressor
    Parameters
    ----------
    max_depth : Profondeur maximale des estimateurs de régression individuels,
    obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in max_depth:
        models[i] = GradientBoostingRegressor(max_depth=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)


# --------------------------------------------------------------------
# GB- GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre min_samples_split
# Le nombre minimum d'échantillons requis pour diviser un nœud interne.
# --------------------------------------------------------------------

def regle_gradboost_minsamplessplit(
        min_samples_split, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre min_samples_leaf de GradientBoostRegressor
    Parameters
    ----------
    min_samples_split : Le nombre minimum d'échantillons requis pour diviser un
    nœud interne., obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in min_samples_split:
        models[i] = GradientBoostingRegressor(
            min_samples_split=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# GB- GRADIENT BOOST REGRESSOR - règle l'hyper-paramètre min_samples_leaf
# Le nombre minimum d'échantillons requis pour se trouver à un nœud de feuille.
# --------------------------------------------------------------------


def regle_gradboost_minsamplesleaf(
        min_samples_leaf, X, y, affiche_boxplot=True):
    """
    Voir l'influence de l'hyperparamètre min_samples_leaf de GradientBoostRegressor
    Parameters
    ----------
    min_samples_leaf : Le nombre minimum d'échantillons requis pour se trouver
    à un nœud de feuille., obligatoire.
    categorical_features_index : les index des variables catégorielles,
    obligatoire
    X : X_train, obligatoire
    y : y_train, obligatoire
    affiche_boxplot : affiche les boxplots? facultatif, défaut True
    Returns
    -------
    Liste du r2 pour chaque nombre d'arbres et graphique boxplot si souhaité.
    """
    models = dict()
    # explore number of features from 1 to 20
    for i in min_samples_leaf:
        models[i] = GradientBoostingRegressor(
            min_samples_leaf=i, random_state=21)

    # Evaluer le modèle avec la valeur de l'hyperparamètre
    evaluer_hyperparametre(models, X, y, affiche_boxplot)

# --------------------------------------------------------------------
# GB- GRADIENT BOOSTING REGRESSOR - RANDOMIZED SEARCH CV
# --------------------------------------------------------------------


def gradientboostingregressor_randomized_search_cv(x_train, y_train, x_test,
                                                   y_test, seed=21):
    '''
    Randomized Search Cv pour le modèle ExtraTreesRegressor
    Parameters
    ----------
    x_train : jeu d'entraînement matrice X, obligatoire
    y_train : jeu d'entraînement target y, obligatoire
    x_test : jeu de test matrice X, obligatoire.
    y_test : jeu de test target y, obligatoire.
    seed : nombre aléatoire pour la reproductibilité des résultats, facultatif
    Returns
    -------
    best_gb_random : Le modèl trouvé aléatoirement avec RandomizedSearchCV
    '''

    # Définition  de la grille
    # ----------------------------------------------------------------------
    # Nombre d'arbres
    n_estimators = [107, 108, 109, 110]
    # Taux d'apprentissge
    learning_rate = [0.05, 0.1]
    # Profondeur de l'arbre
    max_depth = [3, 4, 5, 6]
    # Nombre minimum d'échantillons requis pour chaque split de décision
    min_samples_split = [19, 21]
    # Nombre minimum d'échantillons requis pour chaque feuille
    min_samples_leaf = [1, 2, 4]
    # Grille de recherche
    random_gb_grid = {'n_estimators': n_estimators,
                      'learning_rate': learning_rate,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf}
    print('Grille de recherche : \n')
    pprint(random_gb_grid)

    # Instanciation du modèle
    gb_rand = GradientBoostingRegressor()
    # Recherche aléatoires de paramètres avec 3 folds de cross validation
    # pour 100 itérations avec différentes combinaisons utilisant tous les
    # processeurs
    gb_random = RandomizedSearchCV(estimator=gb_rand,
                                   param_distributions=random_gb_grid,
                                   n_iter=100, cv=3, verbose=0,
                                   random_state=1, n_jobs=-1)
    # Entraînement du modèle
    gb_random.fit(x_train, y_train)

    # Affiche les meilleurs hyperparamètres
    print('\nMeilleurs hyperparamètres : \n')
    pprint(gb_random.best_params_)

    # Evaluation du modèle issu de RandomizedSearchCV
    best_gb_random = gb_random.best_estimator_
    evaluate(best_gb_random, x_test, y_test)

    # Calcul de l'importance de chaque variables
    feature_gb_importance = best_gb_random.feature_importances_
    sorted_idx = feature_gb_importance.argsort()
    feature_gb_importance_tri = outils_data.sort_array(feature_gb_importance)
    # Visualisation des features importance
    plot_features_importance(
        feature_gb_importance_tri,
        x_train.columns[sorted_idx])

    return best_gb_random

# --------------------------------------------------------------------
# GB- GRADIENT BOOSTING REGRESSOR - RANDOMIZED SEARCH CV AVEC GRILLE
# --------------------------------------------------------------------


def gradientboostingregressor_randomizedsearchcv_paramgrid(
        random_gb_grid, gb_rand, x_train, y_train, x_test, y_test, seed=21):
    '''
    Randomized Search Cv pour le modèle ExtraTreesRegressor
    Parameters
    ----------
    random_gb_grid : grille de recherche des hyperparamètres pour RandomizedSearch
    x_train : jeu d'entraînement matrice X, obligatoire
    y_train : jeu d'entraînement target y, obligatoire
    x_test : jeu de test matrice X, obligatoire.
    y_test : jeu de test target y, obligatoire.
    seed : nombre aléatoire pour la reproductibilité des résultats, facultatif
    Returns
    -------
    best_gb_random : Le modèl trouvé aléatoirement avec RandomizedSearchCV
    '''

    # Définition  de la grille
    # ----------------------------------------------------------------------
    print('Grille de recherche : \n')
    pprint(random_gb_grid)

    # Instanciation du modèle
    gb_rand = GradientBoostingRegressor()
    # Recherche aléatoires de paramètres avec 3 folds de cross validation
    # pour 100 itérations avec différentes combinaisons utilisant tous les
    # processeurs
    gb_random = RandomizedSearchCV(estimator=gb_rand,
                                   param_distributions=random_gb_grid,
                                   n_iter=100, cv=3, verbose=0,
                                   random_state=1, n_jobs=-1)
    # Entraînement du modèle
    gb_random.fit(x_train, y_train)

    # Affiche les meilleurs hyperparamètres
    print('\nMeilleurs hyperparamètres : \n')
    pprint(gb_random.best_params_)

    # Evaluation du modèle issu de RandomizedSearchCV
    best_gb_random = gb_random.best_estimator_
    evaluate(best_gb_random, x_test, y_test)

    # Calcul de l'importance de chaque variables
    feature_gb_importance = best_gb_random.feature_importances_
    sorted_idx = feature_gb_importance.argsort()
    feature_gb_importance_tri = outils_data.sort_array(feature_gb_importance)
    # Visualisation des features importance
    plot_features_importance(
        feature_gb_importance_tri,
        x_train.columns[sorted_idx])

    return best_gb_random

# --------------------------------------------------------------------
# GB- GRADIENT BOOSTING REGRESSOR - GRID SEARCH CV - PARAM GRID
# --------------------------------------------------------------------


def gradientboostingregressor_gridsearchcv_paramgrid(
        param_gb_grid, x_train, y_train, x_test, y_test):
    '''
    Grid Search Cv pour le modèle GradientBoostingRegressor
    Parameters
    ----------
    param_gb_grid : grille de recherche des hyperparamètres pour GridSearch CV
    x_train : jeu d'entraînement matrice X, obligatoire
    y_train : jeu d'entraînement target y, obligatoire
    x_test : jeu de test matrice X, obligatoire.
    y_test : jeu de test target y, obligatoire.
    Returns
    -------
    best_gb_grid : Le modèl trouvé aléatoirement avec GridSearchCV
    '''
    print('Grille de recherche : \n')
    pprint(param_gb_grid)

    # Instanciation du modèle
    gb_grid = GradientBoostingRegressor()

    # Instanciation de la recherche sur grille avec validation croisée
    grid_gb_search = GridSearchCV(
        estimator=gb_grid,
        param_grid=param_gb_grid,
        cv=10,
        n_jobs=-1,
        verbose=0)

    # Entraînement de la GridSearchCV
    grid_result_gb = grid_gb_search.fit(x_train, y_train)

    # Sauvegarde du modèle
    with open('modeles/modele_gradientboostRegressor_grid_man.pickle', 'wb') as f:
        pickle.dump(grid_result_gb, f, pickle.HIGHEST_PROTOCOL)

    print(
        f'Best score : {grid_result_gb.best_score_} \n\navec les hyperparamètres :\n{grid_result_gb.best_params_}')

    best_gb_grid = GradientBoostingRegressor(**grid_result_gb.best_params_)

    # Entraînement du modèle
    best_gb_grid.fit(x_train, y_train)

    evaluate(best_gb_grid, x_test, y_test)

    return best_gb_grid

# --------------------------------------------------------------------
# GB- GRADIENT BOOSTING REGRESSOR - GRID SEARCH CV
# --------------------------------------------------------------------


def gradientboostingregressor_grid_search_cv(x_train, y_train, x_test, y_test):
    '''
    Grid Search Cv pour le modèle GradientBoostingRegressor
    Parameters
    ----------
    x_train : jeu d'entraînement matrice X, obligatoire
    y_train : jeu d'entraînement target y, obligatoire
    x_test : jeu de test matrice X, obligatoire.
    y_test : jeu de test target y, obligatoire.
    Returns
    -------
    best_gb_grid : Le modèl trouvé aléatoirement avec GridSearchCV
    '''
    # Grille de paramètre
    param_gb_grid = {'n_estimators': [107, 108],
                     'learning_rate': [0.05, 0.1],
                     'min_samples_leaf': [1, 2, 4],
                     'min_samples_split': [19, 21],
                     'max_depth': [3, 4]
                     }
    print('Grille de recherche : \n')
    pprint(param_gb_grid)

    # Instanciation du modèle
    gb_grid = GradientBoostingRegressor()

    # Instanciation de la recherche sur grille avec validation croisée
    grid_gb_search = GridSearchCV(
        estimator=gb_grid,
        param_grid=param_gb_grid,
        cv=10,
        n_jobs=-1,
        verbose=0)

    # Entraînement de la GridSearchCV
    grid_result_gb = grid_gb_search.fit(x_train, y_train)

    # Sauvegarde du modèle
    with open('modeles/modele_gradientboostRegressor_grid_man.pickle', 'wb') as f:
        pickle.dump(grid_result_gb, f, pickle.HIGHEST_PROTOCOL)

    print(
        f'Best score : {grid_result_gb.best_score_} \n\navec les hyperparamètres :\n{grid_result_gb.best_params_}')

    best_gb_grid = GradientBoostingRegressor(**grid_result_gb.best_params_)

    # Entraînement du modèle
    best_gb_grid.fit(x_train, y_train)

    evaluate(best_gb_grid, x_test, y_test)

    return best_gb_grid

# --------------------------------------------------------------------
# GB- GRADIENT BOOSTING REGRESSOR - GRID SEARCH CV ITERATIVE
# --------------------------------------------------------------------


def gbm_gridsearchcv_iterative(param_grid, model, x_train, y_train):
    '''
    Grid Search Cv pour le modèle ExtraTreesRegressor
    Parameters
    ----------
    param_grid : grille de paramètres, obligatoire
    model : l'algorithme GradientBoostingRegressorà tester, obligatoire
    x_train : jeu d'entraînement matrice X, obligatoire
    y_train : jeu d'entraînement target y, obligatoire
    Returns
    -------
    best_gb_grid : Le modèl trouvé aléatoirement avec GridSearchCV
    '''

    # Instanciation de la recherche sur grille avec validation croisée
    grid_gb_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                  cv=10, n_jobs=-1, verbose=0,
                                  scoring='r2')

    # Entraînement de la GridSearchCV
    grid_result_gb = grid_gb_search.fit(x_train, y_train)

    print(
        f'\nBest score : {grid_result_gb.best_score_} \n\navec les hyperparamètres :\n{grid_result_gb.best_params_}')
    print(f'\nModèle : {grid_result_gb.best_estimator_} \n')


# -----------------------------------------------------------------------
# -- PLOT LES FEATURES IMPORTANCES
# -----------------------------------------------------------------------

def plot_features_importance(features_importance, nom_variables):
    '''
    Affiche le liste des variables avec leurs importances par ordre décroissant.
    Parameters
    ----------
    features_importance: les features importances, obligatoire
    nom_variables : nom des variables, obligatoire
    Returns
    -------
    None.
    '''
    # BarGraph de visalisation
    plt.figure(figsize=(6, 5))
    plt.barh(nom_variables, features_importance)
    plt.xlabel('Feature Importances (%)')
    plt.ylabel('Variables')
    plt.title('Comparison des Features Importances')
    plt.show()

# -----------------------------------------------------------------------
# -- PLOT LES SHAP VALUES
# -----------------------------------------------------------------------


def plot_shape_values(model, x_test):
    '''
    Affiche les SHAPE VALUES.
    Parameters
    ----------
    model: le modèle de machine learning, obligatoire
    x_test :le jeu de test de la matrice X, obligatoire
    Returns
    -------
    None.
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)

    shap.summary_plot(shap_values, x_test, plot_type="bar")

    shap.summary_plot(shap_values, x_test)

    # shap.initjs()
    # shap.force_plot(explainer.expected_value, shap_values[1,:], X_test_log.iloc[1,:])

# -----------------------------------------------------------------------
# -- PLOT LES SHAP VALUES AVEC SKLEARN
# -----------------------------------------------------------------------


def plot_permutation_importance(model, x_test, y_test):
    '''
    Affiche les SHAPE VALUES.
    Parameters
    ----------
    model: le modèle de machine learning, obligatoire
    x_test :le jeu de test de la matrice X, obligatoire
    y_test :le jeu de test de la target, obligatoire
    Returns
    -------
    None.
    '''
    perm_importance = permutation_importance(model, x_test, y_test)

    sorted_idx = perm_importance.importances_mean.argsort()
    plt.figure(figsize=(6, 6))
    plt.barh(x_test.columns[sorted_idx],
             perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance (%)")
    plt.show()

# -----------------------------------------------------------------------
# -- PLOT LES SHAP VALUES AVEC ELI5
# -----------------------------------------------------------------------


def plot_permutation_importance_eli5(model, x_test, y_test):
    '''
    Affiche les SHAPE VALUES.
    Parameters
    ----------
    model: le modèle de machine learning, obligatoire
    x_test :le jeu de test de la matrice X, obligatoire
    y_test :le jeu de test de la target, obligatoire
    Returns
    -------
    None.
    '''
    perm = PermutationImportance(model, random_state=21).fit(x_test, y_test)
    display(eli5.show_weights(perm, feature_names=x_test.columns.tolist()))


# -----------------------------------------------------------------------
# -- EVALUE LE RESSULTAT D'UN MODELE
# -----------------------------------------------------------------------

def evaluate(model, X_test, y_test):
    '''
    Evalue le résultat d'un modèle, MAE, RMSE, R2, accuracy'
    Parameters
    ----------
    model : modèle de machine learning à évaluer, obligatoire
    X_test : jeu de test matrice X, obligatoire
    y_test : jeu de test target y, obligatoire
    Returns
    -------
    accuracy : précision du modèle par rapport aux erreurs.

    '''
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    print('\nPerformance du modèle :\n')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, predictions)
    print(f'mae={mae}')
    print(f'mse={mse}')
    print(f'rmse={rmse}')
    print(f'r2={r2}')

    return accuracy

# -----------------------------------------------------------------------
# -- TRACE LEARNING CURVE POUR VOIR L'OVER ou UNDER FITTING
# -----------------------------------------------------------------------


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Affiche la  learning curve pour je jeu de données de test et d'entraînement
    Parameters
    ----------
    estimator : object qui implemente les méthodes "fit" and "predict"
    title : string, titre du graphique
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training exemples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()


# -----------------------------------------------------------------------
# -- RFE-CV -recursuve feature elimination
# -----------------------------------------------------------------------

def calcul_plot_rfecv(
    estimator,
    model,
    title,
    X_train,
    X_test,
    y_train,
    y_test,
    df_resultats_rfe,
    ylim=None,
    cv=None,
):
    '''
    Effectuer de la Recursive Feature
    Parameters
    ----------
    estimator : modèle réduit par RFECV, obligatoire.
    model : modèle à étudier, obligatoire.
    title : Titre à afficher sur le graphique
    X_train : input du jeu d'entraînement, obligatoire
    X_test : input du jeu de test, obligatoire
    y_train : target du jeu d'entraînement, obligatoire
    y_test : target du jeu de test, obligatoire
    df_resultats_rfe : trace des scores, performance du modèle RFECV réduit
    ylim : limite sur l'axe des ordonnées, optionnel (None par défaut)
    cv : nombre de k-fold pour la cross-validation, (None par défaut).
    Returns
    -------
    df_resultats_rfe : score, trace des scores, performance du modèle RFECV réduit
    y_pred_et_optimise : pérédictions du modèle RFECV.
    model : modèle RFECV.
    '''

    # Traitement des variables hautement corrélées
    # Variables corrélées? Les variables corrélées donnent la même information
    # et doivent être supprimées avant de passer RFECV qui peut être assez long
    correlated_features = set()
    correlation_matrix = X_train.corr()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    print('Variable(s) hautement corrélée(s) à supprimer :\n')
    pprint(correlated_features)

    X_train_rfe = X_train.copy()
    X_train_rfe.drop(correlated_features, 1, inplace=True)

    # RFECV
    selector = RFECV(estimator=estimator, step=1,
                     scoring='neg_mean_squared_error', cv=5, verbose=0)
    selector.fit(X_train_rfe, y_train)

    print(f'\nLe nombre optimal de variables est : {selector.n_features_}')
    features = [f for f, s in zip(X_train_rfe.columns, selector.support_) if s]
    print('\nLes variables sélectionnées sont:')
    pprint(features)

    # Plot RFECV
    plt.figure(figsize=(16, 9))
    plt.title(
        'RFECV : Recursive Feature Elimination with Cross-Validation',
        fontsize=18,
        fontweight='bold',
        pad=20)
    plt.xlabel('Nombres de Variables sélectionnées', fontsize=14, labelpad=20)
    plt.ylabel("Cross validation score (MSE)", fontsize=14, labelpad=20)
    plt.plot(range(1, len(selector.grid_scores_) + 1),
             selector.grid_scores_, color='#303F9F', linewidth=3)

    plt.show()

    # Les variables non indispensables
    print('\nLes variables non indispensables :\n')
    pprint(X_train_rfe.columns[np.where(selector.support_ == False)[0]])
    # Suppression de ces variables
    X_train_rfe.drop(X_train_rfe.columns[np.where(
        selector.support_ == False)[0]], axis=1, inplace=True)

    # Plot importance des variables
    dset = pd.DataFrame()
    dset['variables'] = X_train_rfe.columns
    dset['importance'] = selector.estimator_.feature_importances_

    dset = dset.sort_values(by='importance', ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(y=dset['variables'], width=dset['importance'], color='SteelBlue')
    plt.title('RFECV - Importances des variables',
              fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Importance', fontsize=14, labelpad=20)
    plt.show()

    # Validation du modèle optimisé avec les 23 variables utilisées sur 28
    X_train_reduit = X_train[features]
    X_test_reduit = X_test[features]

    # Modèle ExtraTreesRegressor optimisé entraîné sur jeu réduit
    model.fit(X_train_reduit, y_train)

    # Scoring et prédictions
    df_resultats_rfe, y_pred_et_optimise = process_regression(model,
                                                              X_train_reduit, X_test_reduit,
                                                              y_train, y_test,
                                                              df_resultats_rfe,
                                                              title,
                                                              True, False, False)

    return df_resultats_rfe, y_pred_et_optimise, model


# -----------------------------------------------------------------------
# -- RFE-CV -recursive feature elimination
# -----------------------------------------------------------------------

def calcul_plot_rfecv_cat(estimator, X_train, y_train, ylim=None, cv=None,):
    '''
    RFE-CV -recursive feature elimination spécial CatBoostRegressor à
    cause de l'encodage effectué par CatBoost (categorical_features)
    Parameters
    ----------
    estimator : modèle CatBoostRegressor à étudier
    X_train : input du jeu d'entraînement, obligatoire
    y_train : target du jeu d'entraînement, obligatoire
    ylim : limite sur l'axe des ordonnées, optionnel (None par défaut)
    cv : nombre de k-fold pour la cross-validation, (None par défaut).
    Returns
    -------
    estimator : Modèle RFECV
    '''
    # Traitement des variables hautement corrélées
    # Variables corrélées? Les variables corrélées donnent la même information
    # et doivent être supprimées avant de passer RFECV qui peut être assez long
    correlated_features = set()
    correlation_matrix = X_train.corr()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    print('Variable(s) hautement corrélée(s) à supprimer :\n')
    pprint(correlated_features)

    X_train_rfe = X_train.copy()
    X_train_rfe.drop(correlated_features, 1, inplace=True)

    # RFECV
    selector = RFECV(
        estimator=estimator,
        step=1,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=0)
    selector.fit(X_train_rfe, y_train)

    print(f'\nLe nombre optimal de variables est : {selector.n_features_}')
    features = [f for f, s in zip(X_train_rfe.columns, selector.support_) if s]
    print('\nLes variables sélectionnées sont:')
    pprint(features)

    # Plot RFECV
    plt.figure(figsize=(16, 9))
    plt.title(
        'RFECV : Recursive Feature Elimination with Cross-Validation',
        fontsize=18,
        fontweight='bold',
        pad=20)
    plt.xlabel('Nombres de Variables sélectionnées', fontsize=14, labelpad=20)
    plt.ylabel("Cross validation score (MSE)", fontsize=14, labelpad=20)
    plt.plot(range(1, len(selector.grid_scores_) + 1),
             selector.grid_scores_, color='#303F9F', linewidth=3)

    plt.show()

    # Les variables non indispensables
    print('\nLes variables non indispensables :\n')
    pprint(X_train_rfe.columns[np.where(selector.support_ == False)[0]])
    # Suppression de ces variables
    X_train_rfe.drop(X_train_rfe.columns[np.where(
        selector.support_ == False)[0]], axis=1, inplace=True)

    # Plot importance des variables
    dset = pd.DataFrame()
    dset['variables'] = X_train_rfe.columns
    dset['importance'] = selector.estimator_.feature_importances_

    dset = dset.sort_values(by='importance', ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(y=dset['variables'], width=dset['importance'], color='SteelBlue')
    plt.title('RFECV - Importances des variables',
              fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Importance', fontsize=14, labelpad=20)
    plt.show()

    return estimator


def regression(regressor, x_train, x_test, y_train):
    '''
    Entraînement du modèle et prédictions sur le jeu d'entraînement et le
    jeu de test.
    Parameters
    ----------
    regressor : TYPE
        DESCRIPTION.
    x_train : TYPE
        DESCRIPTION.
    x_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.

    Returns
    -------
    y_train_reg : TYPE
        DESCRIPTION.
    y_test_reg : TYPE
        DESCRIPTION.
    '''
    regressor.fit(x_train, y_train)
    y_train_reg = regressor.predict(x_train)
    y_test_reg = regressor.predict(x_test)

    return y_train_reg, y_test_reg


def scores(regressor, y_train, y_test, y_train_reg, y_test_reg):
    '''
    Génère le score du modèle
    Parameters
    ----------
    regressor : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    y_train_reg : TYPE
        DESCRIPTION.
    y_test_reg : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    ev_train_c = explained_variance_score(y_train, y_train_reg)
    ev_test_c = explained_variance_score(y_test, y_test_reg)
    r2_train_c = r2_score(y_train, y_train_reg)
    r2_test_c = r2_score(y_test, y_test_reg)
    mse_train_c = mean_squared_error(y_train, y_train_reg)
    mse_test_c = mean_squared_error(y_test, y_test_reg)
    mae_train_c = mean_absolute_error(y_train, y_train_reg)
    mae_test_c = mean_absolute_error(y_test, y_test_reg)
    mdae_train_c = median_absolute_error(y_train, y_train_reg)
    mdae_test_c = median_absolute_error(y_test, y_test_reg)
    print(30 * "_")
    print(str(regressor))
    print(30 * "_")
    print("EV score. Train: ", ev_train_c)
    print("EV score. Test: ", ev_test_c)
    print(10 * "-")
    print("R2 score. Train: ", r2_train_c)
    print("R2 score. Test: ", r2_test_c)
    print(10 * "-")
    print("MSE score. Train: ", mse_train_c)
    print("MSE score. Test: ", mse_test_c)
    print(10 * "-")
    print("MAE score. Train: ", mae_train_c)
    print("MAE score. Test: ", mae_test_c)
    print(10 * "-")
    print("MdAE score. Train: ", mdae_train_c)
    print("MdAE score. Test: ", mdae_test_c)
