# -*- coding: utf-8 -*-
""" Librairie personnelle pour manipulation les segmentations de clientèles
"""

# ====================================================================
# Outils Segmentation clientèle - projet 5 Openclassrooms
# Version : 0.0.0 - CRE LR 30/04/2021
# ====================================================================

import pandas as pd
import numpy as np
import datetime as dt
import time
from IPython.display import display
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import folium
# Clustering Metrics
from sklearn.metrics import davies_bouldin_score, silhouette_score, \
    calinski_harabasz_score
# Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
from sklearn import mixture
from kmodes.kprototypes import KPrototypes
import hdbscan
from math import pi
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.cluster.hierarchy import dendrogram
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.cluster import adjusted_rand_score
import statistics
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.0'


# ###########################################################################
# -- PARTIE 1 - SEGMENTATION RFM
# ###########################################################################


# --------------------------------------------------------------------
# -- Segmentation de clientèle à partir d'une date + métrique de stabilité
# --------------------------------------------------------------------

def segmentation_rfm_periode(dataframe, dataframe_rfm_complet,
                             dataframe_resutat, id_unique, var_recence,
                             var_frequence, var_montant, datetime, titre):
    '''
    Segmentation de clientèle à partir d'une date + métrique de stabilité.
    Parameters
    ----------
    dataframe : dataframe à analyser, obligatoire.
    dataframe_rfm_complet : dataframe avec la segmentationRFM sur la période
                           complète d'analyse, obligatoire.
    dataframe_resutat : dataframe de sauvegarde des scores ARI, obligatoire.
    var_recence : date de dernière achat effectué (object), obligatoire.
    var_frequence : nombre d'achat, obligatoire.
    var_montant : montant d'achat, obligatoire.
    datetime : date de fin d'analyse de la segmentation avant la date de fin
               d'historique, string au format 'YYYY-MM-DD HH24:MI:SS'.
    titre : titre correspondant à la période de résultat pour le dataframe de
            sauvegarde des scores ARI, obligatoire.
    Returns
    -------
    dataframe_resutat : dataframe des sauvegarde des résultats ARI
    df_rfm : le dataframe de segmentation RFM sur la période historique.
    '''

    # Conversion de la colonne 'order_purchase_timestamp' au format datetime
    # pour calculs ultérieurs
    dataframe[var_recence] = pd.to_datetime(dataframe[var_recence],
                                            format='%Y-%m-%d %H:%M:%S')

    # Dataframe arrêté à la date transmise en paramètre avant la fin de
    # l'historique
    data = dataframe[dataframe[var_recence] <= datetime]

    # Vérification min max historique
    print('Période : Min : {}, Max : {}'.format(min(data[var_recence]),
                                                max(data[var_recence])))

    # Date de référence fixée au lendemain de la dernière date enregistrée
    # dans la table
    date_reference = max(data[var_recence]) + dt.timedelta(1)
    print('Date de référence : {}'.format(date_reference))

    # Table RFM x mois avant fin historique
    df_rfm = data.groupby(id_unique).agg({var_recence:
                                          lambda x: (date_reference
                                                     - x.max()).days,
                                          var_frequence: 'count',
                                          var_montant: 'sum'}
                                         )
    df_rfm.rename(columns={var_recence: 'rfm_recence',
                           var_frequence: 'rfm_frequence',
                           var_montant: 'rfm_montant'}, inplace=True)

    # Utilisation de la méthode des quantiles
    df_quantiles = df_rfm.quantile(q=[0.25, 0.5, 0.75])
    df_quantiles = df_quantiles.to_dict()

    # Assignation des bins aux variables R, F, M
    # ------------------------------------------------------------------------
    df_rfm['R'] = df_rfm['rfm_recence'] \
        .apply(R_Score, args=('rfm_recence', df_quantiles))
    df_rfm['F'] = df_rfm['rfm_frequence']. \
        apply(FM_Score, args=('rfm_frequence', df_quantiles))
    df_rfm['M'] = df_rfm['rfm_montant'] \
        .apply(FM_Score, args=('rfm_montant', df_quantiles))

    # Concaténation des groupes R, F et M ==> segment du client
    # ------------------------------------------------------------------------
    df_rfm['RFM_Segment'] = [str(row[0]) + str(row[1]) + str(row[2])
                             for row in zip(df_rfm['R'], df_rfm['F'],
                                            df_rfm['M'])]

    # Calul du score
    # ------------------------------------------------------------------------
    df_rfm['RFM_Score'] = df_rfm[['R', 'F', 'M']].sum(axis=1)

    # Ajout d'une variable RFM de classement du type de client
    # ---------------------------------------------------------------------
    score_1 = ['444']
    score_2 = ['331', '332', '333', '334', '341', '342', '343', '431',
               '432', '433', '434', '441', '442', '443']
    score_3 = ['313', '314', '323', '324', '413', '414', '423', '424']
    score_4 = ['411']
    score_5 = ['321', '322', '412', '421', '422']
    score_6 = ['121', '122', '123', '124', '131', '132', '223', '224',
               '231', '232']
    score_7 = ['211', '212', '213', '214', '221', '223', '311', '312']
    score_8 = ['133', '134', '143', '233', '234', '241', '242', '243']
    score_9 = ['344']
    score_10 = ['244']
    score_11 = ['141', '142']
    score_12 = ['144']
    score_13 = ['111', '112', '113', '114']
    df_rfm['RFM_Segm_Client'] = ['Vips Champions'
                                 if row in score_1
                                 else 'Client fidèle'
                                 if row in score_2
                                 else 'Fidélisation potentielle'
                                 if row in score_3
                                 else 'Nouveau client'
                                 if row in score_4
                                 else 'Client prometteur'
                                 if row in score_5
                                 else 'Client ayant besoin attention'
                                 if row in score_6
                                 else 'Client sur le point de dormir'
                                 if row in score_7
                                 else 'Client en danger'
                                 if row in score_8
                                 else 'VIP à ne pas perdre'
                                 if row in score_9
                                 else 'VIP en hibernation'
                                 if row in score_10
                                 else 'Client en hibernation'
                                 if row in score_11
                                 else 'VIP perdu'
                                 if row in score_12
                                 else 'Client perdu'
                                 if row in score_13
                                 else 'Autres'
                                 for row in df_rfm['RFM_Segment']]

    # Replace customer_unique_id comme variable et non index
    df_rfm.reset_index(inplace=True)

    # On garde les clients de cette période historique qui étaient dans le jeu
    # de données de la période complète
    dataframe_rfm_complet = dataframe_rfm_complet[dataframe_rfm_complet.customer_unique_id.isin(
        df_rfm[id_unique])]
    print(dataframe_rfm_complet.shape)
    print(df_rfm.shape)

    # Calcul ARI
    ARI_rfm = metrics.adjusted_rand_score(
        dataframe_rfm_complet.RFM_Segment, df_rfm.RFM_Segment)
    print(f'Métrique ARI : {ARI_rfm}')

    # Ajout ARI dans le tableau de résultats
    dataframe_resutat = dataframe_resutat.append(pd.DataFrame({
        'Periode': [titre],
        'ARI': [ARI_rfm]}), ignore_index=True)

    return dataframe_resutat, df_rfm


# --------------------------------------------------------------------
# -- Découpage de la variable RFM Récence
# --------------------------------------------------------------------
# Groupe des récences
# ------------------------------------------------------------------------
# Fonction permettant de classer chacune des récences dans un des groupes
# de 1 : commande très ancienne à 4 : commande très récente

def R_Score(x, p, d):
    '''
    Transformer une série en 4 groupes de quantiles <0.25 0.25/0.5
    0.5/0.75 et > 0.75
    Parameters
    ----------
    x : Valeur à dispatcher dans une des lignes.
    p : Variable à transformer en groupe de bins de quantile.
    d : dataframe des de la récence en quantile.
    Returns
    -------
    Le groupe à associer :
        - 1 : commande très ancienne
        - 2 : un peu moins ancienne
        - 3 : un peu plus récente
        - et 4 : très récente
    '''
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1  # > 0.75

# --------------------------------------------------------------------
# -- Découpage des variables RFM fréquence et montant
# --------------------------------------------------------------------
# Groupe des fréquences et des montants
# -----------------------------------------------------------------------
# Fonction permettant de classer chacune des fréquences dans un des groupes
# de 1 : commande peu fréquente à 4 : commande très fréquente


def FM_Score(x, p, d):
    '''
    Transformer une série en 4 groupes de quantiles <0.25 0.25/0.5
    0.5/0.75 et > 0.75
    Parameters
    ----------
    x : Valeur à dispatcher dans une des lignes.
    p : Variable à transformer en groupe de bins de quantile.
    d : dataframe des de la récence en quantile.
    Returns
    -------
    Le groupe à associer :
    pour les fréquences :
        - 1 : commande peu fréquente
        - 2 : un peu moins fréquente
        - 3 : un peu plus fréquente
        - et 4 : très fréquente
    pour les montants :
        - 1 : montant bas (faibles dépenses)
        - 2 : un peu moins bas
        - 3 : un peu plus haut
        - et 4 : très haut (fortes dépenses)
    '''
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4  # > 0.75


# --------------------------------------------------------------------
# -- Classement du client en niveau Diamant, Platine, argent, bronze
# --------------------------------------------------------------------

def decouper_clients_niveau(dataframe, variable):
    '''
    Répartition des clients en Diamant, Platine, argent, bronze selon leur
    score.
    Parameters
    ----------
    dataframe : dataframe à analyser, obligatoire.
    variable : variable présentant le score RFM à analyser, obligatoire.
    Returns
    -------
    'Diamant' si score >= 11
    'Platine' si score compris entre 9 et 10
    'Argent' si score compris entre 7 et 8
    'Bronze' si score < 7
    '''
    if dataframe[variable] >= 11:
        return 'Diamant'  # >11
    elif (dataframe[variable] <= 10) and (dataframe[variable] >= 9):
        return 'Platine'  # 9-10
    elif (dataframe[variable] <= 8) and (dataframe[variable] >= 7):
        return 'Argent'  # 7-8
    else:
        return 'Bronze'  # <7


# --------------------------------------------------------------------
# -- Déssine un sankey diagramme avec plusieurs niveaux
# --------------------------------------------------------------------
def genSankey(dataframe, cat_cols=[], value_cols='', title='Sankey Diagram'):
    '''
    Déssine un sankey diagramme avec plusieurs niveaux.
    Parameters
    ----------
    dataframe : dataframe dont on veut afficher sankey diagramme, obligatoire.
    cat_cols : différents niveau de noeuds, optionnel (default is []).
    value_cols : la colonne contenant le nombre d'éléments liés (default is '').
    title : Titre du diagramme, optionnel 'Sankey Diagram'.
    Returns
    -------
    fig : Figure à afficher
    data : data sankey.
    '''
    # maximum of 6 value cols -> 6 colors
    colorPalette = ['#4B8BBE', '#306998', '#FFE873', '#FFD43B', '#646464']
    labelList = []
    colorNumList = []
    for catCol in cat_cols:
        labelListTemp = list(set(dataframe[catCol].values))
        colorNumList.append(len(labelListTemp))
        labelList = labelList + labelListTemp

    # remove duplicates from labelList
    labelList = list(dict.fromkeys(labelList))

    # define colors based on number of levels
    colorList = []
    for idx, colorNum in enumerate(colorNumList):
        colorList = colorList + [colorPalette[idx]] * colorNum

    # transform df into a source-target pair
    for i in range(len(cat_cols) - 1):
        if i == 0:
            sourceTargetDf = dataframe[[
                cat_cols[i], cat_cols[i + 1], value_cols]]
            sourceTargetDf.columns = ['source', 'target', 'count']
        else:
            tempDf = dataframe[[cat_cols[i], cat_cols[i + 1], value_cols]]
            tempDf.columns = ['source', 'target', 'count']
            sourceTargetDf = pd.concat([sourceTargetDf, tempDf])
        sourceTargetDf = sourceTargetDf.groupby(['source', 'target']) \
            .agg({'count': 'sum'}).reset_index()

    # add index for source-target pair
    sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(
        lambda x: labelList.index(x))
    sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(
        lambda x: labelList.index(x))

    # creating the sankey diagram
    data = dict(
        type='sankey',
        node=dict(
            pad=15,
            thickness=20,
            line=dict(
                color="black",
                width=0.5
            ),
            label=labelList,
            color=colorList
        ),
        link=dict(
            source=sourceTargetDf['sourceID'],
            target=sourceTargetDf['targetID'],
            value=sourceTargetDf['count']
        )
    )

    layout = dict(
        title=title,
        font=dict(size=14, color='white'),
        paper_bgcolor='#5B5958'
    )

    fig = dict(data=[data], layout=layout)

    return fig, data


# ###########################################################################
# -- PARTIE 2 - REDUCTION DE DIMENSION
# ###########################################################################


# --------------------------------------------------------------------
# -- SCATTERPLOT DE VISUALISATION DE TRANSFORMATION T-SNE en 2D
# --------------------------------------------------------------------
def affiche_tsne(results_list, liste_param, alpha=0.1):
    '''
    Affiche les résultats de la transformation Lt-SNE
    Parameters
    ----------
    results_list : iste des résultats de la transformation t-SNE, obligatoire.
    liste_param : liste des valeurs de l'hyper paramètre perplexity testées,
                  obligatoire.
    Returns
    -------
    None.
    '''
    i = 0

    # Visualisation en 2D des différents résultats selon la perplexité
    plt.subplots(3, 2, figsize=[15, 20])

    for resultat_tsne in results_list:

        # Perplexité=5
        plt.subplot(3, 2, i + 1)
        tsne_results_i = results_list[i]
        sns.scatterplot(x=tsne_results_i[:, 0], y=tsne_results_i[:, 1],
                        alpha=alpha)
        plt.title('t-SNE avec perplexité=' + str(liste_param[i]))
        plt.grid(False)
        plt.plot()

        i += 1

    plt.show()


# --------------------------------------------------------------------
# -- SCATTERPLOT DE VISUALISATION DE TRANSFORMATION Isomap en 2D
# --------------------------------------------------------------------
def affiche_isomap(X_projected_isomap, titre='Isomap', alpha=1):
    '''
    Affiche le résultat en 2D de la transformation Isomap
    Parameters
    ----------
    X_projected_isomap : résultats de la transformation isomap, obligatoire.
    titre : titre de la figure, optionnel (défaut = Isomap).
    alpha : alpha, optionnel (défaut = 1).
    Returns
    -------
    None.
    '''

    # Visualisation en 2D de la transformation Isomap
    plt.figure(figsize=[6, 6])

    sns.scatterplot(x=X_projected_isomap[:, 0], y=X_projected_isomap[:, 1],
                    alpha=alpha)
    plt.title(titre)
    plt.grid(False)
    plt.show()

# --------------------------------------------------------------------
# -- SCATTERPLOT DE VISUALISATION DE TRANSFORMATION UMAP en 2D
# --------------------------------------------------------------------


def affiche_umap(X_projected_umap, titre='UMAP', alpha=1):
    '''
    Affiche le résultat en 2D de la transformation UMAP
    Parameters
    ----------
    X_projected_umap : résultats de la transformation umap, obligatoire.
    titre : titre de la figure, optionnel (défaut = Isomap).
    alpha : alpha, optionnel (défaut = 1).
    Returns
    -------
    None.
    '''

    # Visualisation en 2D de la transformation Isomap
    plt.figure(figsize=[6, 6])

    sns.scatterplot(x=X_projected_umap[:, 0], y=X_projected_umap[:, 1],
                    alpha=alpha)
    plt.title(titre)
    plt.grid(False)
    plt.show()

# --------------------------------------------------------------------
# -- SCATTERPLOT DE VISUALISATION DE TRANSFORMATION UMAP en 2D
# --------------------------------------------------------------------


def affiche_umap_cat(X_projected_umap, titre='UMAP', alpha=1):
    '''
    Affiche le résultat en 2D de la transformation UMAP en associant les
    variables quantitatives et les variables qualitatives
    Parameters
    ----------
    X_projected_umap : résultats de la transformation umap, obligatoire.
    titre : titre de la figure, optionnel (défaut = Isomap).
    alpha : alpha, optionnel (défaut = 1).
    Returns
    -------
    None.
    '''
    # Visualisation en 2D de la transformation Isomap
    plt.figure(figsize=[20, 15])

    sns.scatterplot(x=X_projected_umap[:, 0], y=X_projected_umap[:, 1],
                    s=20, alpha=alpha)
    plt.title(titre, fontsize=16)
    plt.grid(False)
    plt.show()


# ###########################################################################
# -- PARTIE 3 - CLUSTERING KMeans
# ###########################################################################


# --------------------------------------------------------------------
# -- CALCUL DES METRIQUES K-Means
# --------------------------------------------------------------------

def calcul_metrics_kmeans(data, dataframe_metrique, type_donnees,
                          random_seed, n_clusters, n_init, init):
    '''
    Calcul des métriques de KMeans en fonction de différents paramètres.
    Parameters
    ----------
    data : données, obligatoire.
    dataframe_metrique : dataframe de sauvegarde des résultats, obligatoire.
    type_donnees : string intitulé des données, obligatoire.
    random_seed : nombre aléatoire pour la reproductibilité, obligatoire.
    n_clusters : liste du nombre de clusters, obligatoire,
    n_init : nombre de clusters à initialiser, obligatoire.
    init : type d'initialisation : 'k-means++' ou 'random'.
    Returns
    -------
    dataframe_metrique : résultat des métriques
    '''
    # Cette fonction permet de calculer le nombre de clusters le plus
    # optimal pour notre analyse : le coefficient de silhouette,
    # dispersion, indice de davies_bouldin
    silhouette = []
    dispersion = []
    davies_bouldin = []
    donnees = []
    temps = []

    result_clusters = []
    result_ninit = []
    result_type_init = []

    # Hyperparametre tuning
    n_clusters = n_clusters
    nbr_init = n_init
    type_init = init

    # Tester le modèle entre 2 et 12 clusters
    for num_clusters in n_clusters:

        for init in nbr_init:

            for i in type_init:
                # Top début d'exécution
                time_start = time.time()

                # Initialisation de l'algorithme
                cls = KMeans(n_clusters=num_clusters,
                             n_init=init,
                             init=i,
                             random_state=random_seed)

                # Entraînement de l'algorithme
                cls.fit(data)

                # Prédictions
                preds = cls.predict(data)

                # Top fin d'exécution
                time_end = time.time()

                # Calcul du score de coefficient de silhouette
                silh = silhouette_score(data, preds)
                # Calcul la dispersion
                disp = cls.inertia_
                # Calcul de l'indice davies-bouldin
                db = davies_bouldin_score(data, preds)
                # Durée d'exécution
                time_execution = time_end - time_start

                silhouette.append(silh)
                dispersion.append(disp)
                davies_bouldin.append(db)
                donnees.append(type_donnees)
                temps.append(time_execution)

                result_clusters.append(num_clusters)
                result_ninit.append(init)
                result_type_init.append(i)

    dataframe_metrique = dataframe_metrique.append(pd.DataFrame({
        'Type_données': donnees,
        'nbr_clusters': result_clusters,
        'n_init': result_ninit,
        'type_init': result_type_init,
        'coef_silh': silhouette,
        'dispersion': dispersion,
        'davies_bouldin': davies_bouldin,
        'Durée (s)': temps
    }), ignore_index=True)

    return dataframe_metrique

# ###########################################################################
# -- PARTIE 4 - CLUSTERING CAH Classification Hiérarchique Ascendante
# ###########################################################################


# --------------------------------------------------------------------
# -- CALCUL DES METRIQUES CAH classification hiérarchique ascendante
# --------------------------------------------------------------------

def calcul_metrics_cah(data, dataframe_metrique, type_donnees,
                       random_seed, param_grid):
    '''
    Calcul des métriques de CAH en fonction de différents paramètres.
    Parameters
    ----------
    data : données, obligatoire.
    dataframe_metrique : dataframe de sauvegarde des résultats, obligatoire.
    type_donnees : string intitulé des données, obligatoire.
    random_seed : nombre aléatoire pour la reproductibilité, obligatoire.
    param_grid : la grille contenant les paramètres à optimiser.
    Returns
    -------
    dataframe_metrique : résultat des métriques
    '''
    # Cette fonction permet de calculer le nombre de clusters le plus
    # optimal pour notre analyse : le coefficient de silhouette,
    # indice de davies_bouldin, indice de Calinsky-Harabasz
    silhouette = []
    davies_bouldin = []
    calin_harab = []
    donnees = []
    temps = []

    result_nclusters = []
    result_linkage = []
    result_affinity = []

    # Hyperparametre tuning
    n_clusters = param_grid[0]
    linkage = param_grid[1]
    affinity = param_grid[2]

    # Tester le modèle avec les différentes combinaisons de paramètres
    for nb_clusters in n_clusters:

        for link in linkage:

            for affinite in affinity:

                # Top début d'exécution
                time_start = time.time()

                # Initialisation de l'algorithme
                cah = AgglomerativeClustering(n_clusters=nb_clusters,
                                              linkage=link,
                                              affinity=affinite)

                # Entraînement de l'algorithme / Prédictions
                preds = cah.fit_predict(data)

                # Top fin d'exécution
                time_end = time.time()

                # Calcul du score de coefficient de silhouette
                silh = silhouette_score(data, preds)
                # Calcul de l'indice davies-bouldin
                db = davies_bouldin_score(data, preds)
                # Calcul de l'indice  Calinski_harabasz
                cal_har = calinski_harabasz_score(data, preds)
                # Durée d'exécution
                time_execution = time_end - time_start

                silhouette.append(silh)
                davies_bouldin.append(db)
                calin_harab.append(cal_har)
                donnees.append(type_donnees)
                temps.append(time_execution)

                result_nclusters.append(nb_clusters)
                result_linkage.append(link)
                result_affinity.append(affinite)

    dataframe_metrique = dataframe_metrique.append(pd.DataFrame({
        'Type_données': donnees,
        'n_clusters': result_nclusters,
        'linkage': result_linkage,
        'affinity': result_affinity,
        'coef_silh': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calin_harab,
        'Durée (s)': temps
    }), ignore_index=True)

    return dataframe_metrique


# ###########################################################################
# -- PARTIE 5 - CLUSTERING HIERARCHIQUE DENDOGRAMME AVEC SCIPY
# ###########################################################################


# --------------------------------------------------------------------
# -- AFFICHE DENDOGRAMME AVEC SCIPY
# --------------------------------------------------------------------

def plot_dendrogram(Z, p=10, names=[]):
    '''Affiche un dendogramme des données.
    Parameters:
    ----------
        - Z : linkage. Ex: Z = linkage(X_std, 'ward')
        - labels : noms des clusters
        - p : nombre de clusters maximal à afficher (stop à ce chiffre).
    '''

    sns.set_style('white')

    # Visualisation du dendogramme à p clusters finaux
    plt.figure(figsize=(10, 25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(Z, labels=names, orientation='left')
    plt.show()

    # Visualisation du dendogramme réduit à p clusters
    plt.figure(figsize=(15, 7))
    plt.title('Hierarchical Clustering Dendrogram - Réduit')
    plt.xlabel('Clients')
    plt.ylabel('Inertie')
    dendrogram(Z, p=p, truncate_mode='lastp')
    plt.show()


# ###########################################################################
# -- PARTIE 6 - CLUSTERING MODELISATION A BASE DE DENSITE - DBSCAN
# ###########################################################################

# --------------------------------------------------------------------
# -- Recherche du meilleur paramètre epsilon
# -- Source : https://medium.com/mlearning-ai/demonstrating-
#    customers-segmentation-with-dbscan-clustering-using-python-8a2ba0db2a2e
# --------------------------------------------------------------------
def recherche_epsilon(data, minpts):
    '''
    Recherche du meilleur paramètre epsilon pour paramétrage DBSCAN.
    Parameters
    ----------
    data : données, obligatoire..
    minpts : minimum de points de voisinage dans le rayon epsilon.
    Returns
    -------
    None.
    '''
    neigh = NearestNeighbors(n_neighbors=minpts)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    plt.figure(figsize=(10, 8))
    plt.plot(distances)
    plt.xlabel("Nbr clusters", size=14)
    plt.ylabel("Epsilon", size=14)
    plt.title("Evaluation Epsilon", size=14)


# --------------------------------------------------------------------
# -- CALCUL DES METRIQUES DBSCAN Modélisation à base de densité
# --------------------------------------------------------------------

def calcul_metrics_dbscan(data, dataframe_metrique, type_donnees,
                          random_seed, param_grid):
    '''
    Calcul des métriques de DBSCAN en fonction de différents paramètres.
    Parameters
    ----------
    data : données, obligatoire.
    dataframe_metrique : dataframe de sauvegarde des résultats, obligatoire.
    type_donnees : string intitulé des données, obligatoire.
    random_seed : nombre aléatoire pour la reproductibilité, obligatoire.
    param_grid : la grille contenant les paramètres à optimiser.
    Returns
    -------
    dataframe_metrique : résultat des métriques
    '''
    # Cette fonction permet de calculer le nombre de clusters le plus
    # optimal pour notre analyse : le coefficient de silhouette,
    # indice de davies_bouldin, indice de Calinsky-Harabasz
    silhouette = []
    davies_bouldin = []
    calin_harab = []
    donnees = []
    temps = []

    result_eps = []
    result_minsamples = []

    # Hyperparametre tuning
    eps = param_grid[0]
    min_samples = param_grid[1]
    n_jobs = param_grid[2]

    # Tester le modèle avec les différentes combinaisons de paramètres
    for epsilon in eps:

        for minsamples in min_samples:

            # Top début d'exécution
            time_start = time.time()

            # Initialisation de l'algorithme
            dbscan = DBSCAN(eps=epsilon,
                            min_samples=minsamples,
                            n_jobs=n_jobs)

            # Entraînement de l'algorithme / Prédictions
            preds = dbscan.fit_predict(data)

            # Top fin d'exécution
            time_end = time.time()

            # Calcul du score de coefficient de silhouette
            silh = silhouette_score(data, preds)
            # Calcul de l'indice davies-bouldin
            db = davies_bouldin_score(data, preds)
            # Calcul de l'indice  Calinski_harabasz
            cal_har = calinski_harabasz_score(data, preds)
            # Durée d'exécution
            time_execution = time_end - time_start

            silhouette.append(silh)
            davies_bouldin.append(db)
            calin_harab.append(cal_har)
            donnees.append(type_donnees)
            temps.append(time_execution)

            result_eps.append(epsilon)
            result_minsamples.append(minsamples)

    dataframe_metrique = dataframe_metrique.append(pd.DataFrame({
        'Type_données': donnees,
        'eps': result_eps,
        'min_samples': result_minsamples,
        'coef_silh': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calin_harab,
        'Durée (s)': temps
    }), ignore_index=True)

    return dataframe_metrique


# ###########################################################################
# -- PARTIE 7 - CLUSTERING MODELISATION A BASE DE DENSITE - HDBSCAN
# ###########################################################################


# --------------------------------------------------------------------
# -- CALCUL DES METRIQUES HDBSCAN Modélisation à base de densité
# --------------------------------------------------------------------

def calcul_metrics_hdbscan(data, dataframe_metrique, type_donnees,
                           random_seed, param_grid, method):
    '''
    Calcul des métriques de HDBSCAN en fonction de différents paramètres.
    Parameters
    ----------
    data : données, obligatoire.
    dataframe_metrique : dataframe de sauvegarde des résultats, obligatoire.
    type_donnees : string intitulé des données, obligatoire.
    random_seed : nombre aléatoire pour la reproductibilité, obligatoire.
    param_grid : la grille contenant les paramètres à optimiser.
    methode : oem ou leaf.
    Returns
    -------
    dataframe_metrique : résultat des métriques
    '''
    # Cette fonction permet de calculer le nombre de clusters le plus
    # optimal pour notre analyse : le coefficient de silhouette,
    # indice de davies_bouldin, indice de Calinsky-Harabasz
    silhouette = []
    davies_bouldin = []
    calin_harab = []
    donnees = []
    temps = []

    result_minsamples = []
    result_minclustersize = []

    # Hyperparametre tuning
    min_samples = param_grid[0]
    min_cluster_size = param_grid[1]

    # Tester le modèle avec les différentes combinaisons de paramètres
    for minsamples in min_samples:

        for minclustersize in min_cluster_size:

            # print(minsamples, minclustersize)
            # Top début d'exécution
            time_start = time.time()

            # Initialisation de l'algorithme
            clus_hdbscan = hdbscan.HDBSCAN(algorithm='best', alpha=1.0,
                                           approx_min_span_tree=True,
                                           gen_min_span_tree=True,
                                           leaf_size=40,
                                           cluster_selection_method=method,
                                           metric='euclidean',
                                           min_cluster_size=minclustersize,
                                           min_samples=minsamples,
                                           allow_single_cluster=False,
                                           core_dist_n_jobs=-1)

            # Entraînement de l'algorithme / Prédictions
            preds = clus_hdbscan.fit_predict(data)

            # Top fin d'exécution
            time_end = time.time()

            # Calcul du score de coefficient de silhouette
            silh = silhouette_score(data, preds)
            # Calcul de l'indice davies-bouldin
            db = davies_bouldin_score(data, preds)
            # Calcul de l'indice  Calinski_harabasz
            cal_har = calinski_harabasz_score(data, preds)
            # Durée d'exécution
            time_execution = time_end - time_start

            silhouette.append(silh)
            davies_bouldin.append(db)
            calin_harab.append(cal_har)
            donnees.append(type_donnees)
            temps.append(time_execution)

            result_minsamples.append(minsamples)
            result_minclustersize.append(minclustersize)

    dataframe_metrique = dataframe_metrique.append(pd.DataFrame({
        'Type_données': donnees,
        'min_samples': result_minsamples,
        'min_cluster_size': result_minclustersize,
        'coef_silh': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calin_harab,
        'Durée (s)': temps
    }), ignore_index=True)

    return dataframe_metrique


def calcul_metrics_hdbscan_best(dataframe, method, list_name):
    '''
    Calcul les métriques de hdbscan, hyperparamètre tunning.
    Parameters
    ----------
    dataframe : dataframe à optimiser, obligatoire.
    method : méthode : 'leaf' ou 'oem'.
    list_name : List de sauvegarde des meilleurs paramètres.
    Returns
    -------
    None.

    '''
    global data_leaf, data_eom
    n = dataframe.shape[0]

    for gamma in range(1, int(np.log(n))):

        for ms in range(1, int(2 * np.log(n))):

            clust_alg = hdbscan.HDBSCAN(
                algorithm='best',
                alpha=1.0,
                approx_min_span_tree=True,
                gen_min_span_tree=True,
                leaf_size=40,
                cluster_selection_method=method,
                metric='euclidean',
                min_cluster_size=int(
                    gamma * np.sqrt(n)),
                min_samples=ms,
                allow_single_cluster=False).fit(dataframe)

            min_cluster_size = clust_alg.min_cluster_size
            min_samples = clust_alg.min_samples
            validity_score = clust_alg.relative_validity_
            n_clusters = np.max(clust_alg.labels_)
            list_name.append(
                (min_cluster_size,
                 min_samples,
                 validity_score,
                 n_clusters))

            if validity_score >= .5:
                print(
                    f'min_cluster_size = {min_cluster_size},  min_samples = {min_samples}, validity_score = {validity_score} n_clusters = {n_clusters}')


def best_hdbscan_validity(source):
    '''
    Affiche les meilleurs résultats pour source = méthode leaf ou oem.
    Parameters
    ----------
    source : data_leaf ou data_oem (résultats de calcul_metrics_hdbscan)
    Returns
    -------
    best_validity : la valeur des meilleurs hyperparamètres et le nombre de
                    clusters
    '''
    cols = ['min_cluster_size', 'min_samples', 'validity_score', 'n_clusters']
    dataframe = pd.DataFrame(source, columns=cols)
    best_validity = dataframe.loc[dataframe['validity_score'].idxmax()]

    return best_validity


# ###########################################################################
# -- PARTIE 8 - CLUSTERING MODELISATION A BASE DE DENSITE - OPTICS
# ###########################################################################


def calcul_metrics_optics(data, dataframe_metrique, type_donnees,
                          random_seed, param_grid):
    '''
    Calcul des métriques de OPTICS en fonction de différents paramètres.
    Parameters
    ----------
    data : données, obligatoire.
    dataframe_metrique : dataframe de sauvegarde des résultats, obligatoire.
    type_donnees : string intitulé des données, obligatoire.
    random_seed : nombre aléatoire pour la reproductibilité, obligatoire.
    param_grid : la grille contenant les paramètres à optimiser.
    Returns
    -------
    dataframe_metrique : résultat des métriques
    '''
    # Cette fonction permet de calculer le nombre de clusters le plus
    # optimal pour notre analyse : le coefficient de silhouette,
    # indice de davies_bouldin, indice de Calinsky-Harabasz
    silhouette = []
    davies_bouldin = []
    calin_harab = []
    donnees = []
    temps = []

    result_minsamples = []
    result_xi = []
    result_minclustersize = []

    # Hyperparametre tuning
    min_samples = param_grid[0]
    xi = param_grid[1]
    min_cluster_size = param_grid[2]

    # Tester le modèle avec les différentes combinaisons de paramètres
    for minsamples in min_samples:

        for xi_work in xi:

            for minclustersize in min_cluster_size:

                # display(minsamples, xi_work, minclustersize)

                # Top début d'exécution
                time_start = time.time()

                # Initialisation de l'algorithme
                optics = OPTICS(min_samples=minsamples,
                                xi=xi_work,
                                min_cluster_size=minclustersize
                                )

                # Entraînement de l'algorithme / Prédictions
                preds = optics.fit_predict(data)

               # Top fin d'exécution
                time_end = time.time()

                if len(set(preds)) > 1:

                    # Calcul du score de coefficient de silhouette
                    silh = silhouette_score(data, preds)
                    # Calcul de l'indice davies-bouldin
                    db = davies_bouldin_score(data, preds)
                    # Calcul de l'indice  Calinski_harabasz
                    cal_har = calinski_harabasz_score(data, preds)

                    silhouette.append(silh)
                    davies_bouldin.append(db)
                    calin_harab.append(cal_har)

                else:

                    silhouette.append(0)
                    davies_bouldin.append(00)
                    calin_harab.append(0)

                # Durée d'exécution
                time_execution = time_end - time_start

                donnees.append(type_donnees)
                temps.append(time_execution)

                result_minsamples.append(minsamples)
                result_xi.append(xi_work)
                result_minclustersize.append(minclustersize)

    dataframe_metrique = dataframe_metrique.append(pd.DataFrame({
        'Type_données': donnees,
        'min_samples': result_minsamples,
        'xi': result_xi,
        'min_cluster_size': result_minclustersize,
        'coef_silh': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calin_harab,
        'Durée (s)': temps
    }), ignore_index=True)

    return dataframe_metrique


# ###########################################################################
# -- PARTIE 9 - CLUSTERING MODELISATION AVEC MELANGES GAUSSIENNES
# ###########################################################################


def calcul_metrics_gmm(data, dataframe_metrique, type_donnees,
                       random_seed, param_grid):
    '''
    Calcul des métriques de GaussianMixture en fonction de différents paramètres.
    Parameters
    ----------
    data : données, obligatoire.
    dataframe_metrique : dataframe de sauvegarde des résultats, obligatoire.
    type_donnees : string intitulé des données, obligatoire.
    random_seed : nombre aléatoire pour la reproductibilité, obligatoire.
    param_grid : la grille contenant les paramètres à optimiser.
    Returns
    -------
    dataframe_metrique : résultat des métriques
    '''
    # Cette fonction permet de calculer le nombre de clusters le plus
    # optimal pour notre analyse : le coefficient de silhouette,
    # indice de davies_bouldin, indice de Calinsky-Harabasz
    silhouette = []
    davies_bouldin = []
    calin_harab = []
    donnees = []
    temps = []

    result_ncomponents = []
    result_covtype = []

    # Hyperparametre tuning
    n_components = param_grid[0]
    covariance_type = param_grid[1]

    # Tester le modèle avec les différentes combinaisons de paramètres
    for ncomponents in n_components:

        for covtype in covariance_type:

            # display(ncomponents, covtype)

            # Top début d'exécution
            time_start = time.time()

            # Initialisation de l'algorithme
            gmm = mixture.GaussianMixture(n_components=ncomponents,
                                          covariance_type=covtype)

            # Entraînement de l'algorithme / Prédictions
            preds = gmm.fit_predict(data)

            # Top fin d'exécution
            time_end = time.time()

            # Calcul du score de coefficient de silhouette
            silh = silhouette_score(data, preds)
            # Calcul de l'indice davies-bouldin
            db = davies_bouldin_score(data, preds)
            # Calcul de l'indice  Calinski_harabasz
            cal_har = calinski_harabasz_score(data, preds)
            # Durée d'exécution
            time_execution = time_end - time_start

            silhouette.append(silh)
            davies_bouldin.append(db)
            calin_harab.append(cal_har)
            donnees.append(type_donnees)
            temps.append(time_execution)

            result_ncomponents.append(ncomponents)
            result_covtype.append(covtype)

    dataframe_metrique = dataframe_metrique.append(pd.DataFrame({
        'Type_données': donnees,
        'n_components': result_ncomponents,
        'covariance_type': result_covtype,
        'coef_silh': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calin_harab,
        'Durée (s)': temps
    }), ignore_index=True)

    return dataframe_metrique


# ###########################################################################
# -- PARTIE 10 - CLUSTERING VISUALISATION
# ###########################################################################


# --------------------------------------------------------------------
# -- Affiche la répartition du nombre de clients par clusters
# --------------------------------------------------------------------

def affiche_clients_par_clusters(clusters_labels):
    '''
    Affiche la répartitionn des clients par cluster
    Parameters
    ----------
    clusters_labels : la séries des labels des clusters, obligatoire.
    Returns
    -------
    None.
    '''
    ax1 = plt.gca()

    # DataFrame de travail
    series_client_cluster = pd.Series(clusters_labels).value_counts()
    nb_client = series_client_cluster.sum()
    df_visu_client_cluster = pd.DataFrame(
        {'Clusters': series_client_cluster.index,
         'Nb_clients': series_client_cluster.values})
    df_visu_client_cluster['%'] = round(
        (df_visu_client_cluster['Nb_clients']) * 100 / nb_client, 2)
    df_visu_client_cluster = df_visu_client_cluster.sort_values(by='Clusters')
    display(df_visu_client_cluster.style.hide_index())

    # Barplot de la distribution
    sns.set_style('white')
    sns.barplot(x='Clusters', y='Nb_clients',
                data=df_visu_client_cluster, color='SteelBlue', ax=ax1)
    ax1.set_ylabel('Nombre de clients)', fontsize=12)
    ax1.set_xlabel('Clusters', fontsize=12)
    ax1.set_title('Nombre de clients par clusters', fontsize=14)
    plt.gcf().set_size_inches(6, 4)
    plt.grid(False)
    plt.show()

# --------------------------------------------------------------------
# -- Affiche une visualisation 2D des clients par clusters par t-SNE
# --------------------------------------------------------------------


def affiche_tsne_par_clusters(proj_tsne, clusters_labels, nbre_clusters):
    '''
    Affiche la projection 2D par t-SNE par clusters.
    Parameters
    ----------
    proj_tsne : projection t-SNE, obligatoire.
    clusters_labels : la variable représentant les clusters.
    nbre_clusters : nombre de clusters pour les couleurs
    Returns
    -------
    None.
    '''
    plt.figure(figsize=[12, 12])
    sns.set_style('white')

    if nbre_clusters <= 10:
        sns.scatterplot(proj_tsne[:, 0],
                        proj_tsne[:, 1],
                        hue=clusters_labels,
                        legend='full',
                        palette=sns.color_palette('hls', nbre_clusters))
    else:
        sns.scatterplot(proj_tsne[:, 0],
                        proj_tsne[:, 1],
                        hue=clusters_labels,
                        legend='full')
    plt.show()

# --------------------------------------------------------------------
# -- Affiche une visualisation 2D des clients par clusters par UMAP
# --------------------------------------------------------------------


def affiche_umap_par_clusters(proj_umap, clusters_labels, nbre_clusters):
    '''
    Affiche la projection 2D par UMAP par clusters.
    Parameters
    ----------
    proj_tsne : projection UMAP, obligatoire.
    clusters_labels : la variable représentant les clusters.
    nbre_clusters : nombre de clusters pour les couleurs
    Returns
    -------
    None.
    '''
    plt.figure(figsize=[12, 12])
    sns.set_style('white')

    if nbre_clusters <= 10:
        sns.scatterplot(proj_umap[:, 0],
                        proj_umap[:, 1],
                        hue=clusters_labels,
                        legend='full',
                        palette=sns.color_palette('hls', n_colors=nbre_clusters))
    else:
        sns.scatterplot(proj_umap[:, 0],
                        proj_umap[:, 1],
                        hue=clusters_labels,
                        legend='full')
    plt.show()


# --------------------------------------------------------------------
# -- INTERPRETATION VISUELLE DES CLUSTERS - BARPLOT variables/variables
# --------------------------------------------------------------------

def affiche_variables_par_clusters(dataframe, var_cluster):
    '''
    Affiche les moyennes des variables par clusters.
    Parameters
    ----------
    dataframe : dataframe à interpréter, obligatoire
    Returns
    -------
    None.
    '''

    cols = dataframe.columns.to_list()
    cols.remove(var_cluster)

    sns.set_style('white')

    for col in cols:
        g = sns.catplot(data=dataframe, kind='bar', x=var_cluster, y=col,
                        ci='sd', alpha=.6, height=5, palette='bright')
        g.set_axis_labels("", col)
        plt.title(col, fontsize=20)
        plt.show()

# --------------------------------------------------------------------
# -- INTERPRETATION VISUELLE DES CLUSTERS : snakeplot des clusters
# --------------------------------------------------------------------


def affiche_snakeplot_par_clusters(dataframe, var_cluster):
    '''
    Affiche le snakeplot des variables des différents clusters.
    Parameters
    ----------
    dataframe : dataframe à interpréter, obligatoire.
    var_cluster : variable cluster à analyser.
    Returns
    -------
    None.
    '''
    # Melt du dataframe
    cols_a_interpreter = dataframe.columns.to_list()
    cols_a_interpreter.remove('dernier_achat_annee')
    dataframe_melted = pd.melt(frame=dataframe[cols_a_interpreter],
                               id_vars=[var_cluster],
                               var_name='Metrics', value_name='Value')

    # Visualisation : Snake plot des différentes variables moyennes
    # --------------------------------------------------------------------

    plt.figure(figsize=(20, 10))

    sns.set_style('white')

    sns.lineplot(x='Metrics', y='Value', hue=var_cluster,
                 data=dataframe_melted, palette='bright')
    plt.title('Snake Plot des différents clusters', fontsize='18')
    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper right')

    plt.show()

# --------------------------------------------------------------------
# -- INTERPRETATION VISUELLE DES CLUSTERS : radarplot des clusters
# --------------------------------------------------------------------


def affiche_radarplot_par_clusters(dataframe, var_cluster, nb_rows, nb_cols):
    '''
    Affiche les radars plot des différents clusters pour l'interprétation.
    Parameters
    ----------
    dataframe : dataframe des données, obligatoire.
    var_cluster : str nom de la variable représentant le label des clusters,
                  obligatoire.
    nb_rows : nombre de lignes pour afficher les radars plots, obligatoire.
    nb_cols : nombre de colonnes pour afficher les radars plots.
    Returns
    -------
    None.
    '''
    # Radarplot des différents clusters
    # --------------------------------------------------------------------

    cols_a_interpreter = dataframe.columns.to_list()
    # cols_a_interpreter.remove('dernier_achat_annee')
    df_plot_seg = dataframe[cols_a_interpreter].set_index(var_cluster)
    # Standardisation
    min_max = MinMaxScaler()
    df_plot_seg = pd.DataFrame(min_max.fit_transform(df_plot_seg.values),
                               index=df_plot_seg.index,
                               columns=df_plot_seg.columns)
    plt.rc('axes', facecolor='Gainsboro')

    # number of variable
    categories = list(df_plot_seg.columns)

    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot /
    # number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]

    fig = plt.figure(1, figsize=(20, 35))

    colors = ['#023eff', '#ff7c00', '#1ac938', '#e8000b', '#892adf',
              '#9f4800', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff']

    for i, cluster in enumerate(df_plot_seg.index):

        ax = fig.add_subplot(nb_rows, nb_cols, i + 1, polar=True)

        ax.set_theta_offset(2 * pi / 3)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles, categories, size=10)
        plt.yticks(color="grey", size=2)
        values = df_plot_seg.iloc[i].values
        ax.plot(angles, values, 'o-', linewidth=1, linestyle='solid')
        ax.fill(angles, values, colors[i], alpha=0.55)
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_title('Cluster ' + str(cluster), size=20, color=colors[i])
        ax.grid(True)
        plt.grid(True)
        plt.ylim(0, 1)

    plt.show()

# --------------------------------------------------------------------
# -- Affiche les graphiques des différents scores des clusters
# --------------------------------------------------------------------


def affiche_effet_clusters_score(dataframe, nclust_deb, nclust_fin, silh=True,
                                 davbou=True, calhar=True):
    '''
    Affiche les graphiques des différents scores des clusters.
    Parameters
    ----------
    dataframe : dataframe de résultats des métriques, obligatoire.
    nclust_deb : nombre de clusters minimum, obligatoire.
    nclust_fin : nombre de clusters maximum, obligatoire.
    silh : Affiche graphique score des silhouettes?, optionnel (défaut : True)
    davbou : Affiche graphique des indices de Davies-Bouldin?,
             optionnel (défaut : True)
    calhar : Affiche graphique des indices de Calinski-Harabasz?,
             optionnel (défaut : True)
    Returns
    -------
    None.
    '''
    if silh:
        # Trace le graphique des coefficient de silhouette
        plt.figure(figsize=(12, 5))

        s_silhouette = dataframe.groupby('n_clusters')['coef_silh'].mean()

        plt.title('Le coefficient de silhouette', fontsize=16)
        plt.plot([i for i in range(nclust_deb, nclust_fin)], s_silhouette,
                 marker='o')
        plt.grid(True)
        plt.xlabel('Nombre de groupes', fontsize=14)
        plt.ylabel('Silhouette score', fontsize=15)
        plt.xticks([i for i in range(nclust_deb, nclust_fin)], fontsize=14)
        plt.yticks(fontsize=15)

        plt.show()

    if davbou:
        # Trace le graphique des indices de davies-bouldin
        plt.figure(figsize=(12, 5))

        s_db = dataframe.groupby('n_clusters')['davies_bouldin'].mean()

        plt.title('L\'indice de Davies-Bouldin', fontsize=16)
        plt.plot([i for i in range(nclust_deb, nclust_fin)], s_db, marker='o')

        plt.grid(True)
        plt.xlabel('Nombre de groupes', fontsize=14)
        plt.ylabel('Indice de Davies-Bouldin', fontsize=15)
        plt.xticks([i for i in range(nclust_deb, nclust_fin)], fontsize=14)
        plt.yticks(fontsize=15)

        plt.show()

    if calhar:
        # Trace le graphique des indices de Calinski-Harabasz
        plt.figure(figsize=(12, 5))

        s_db = dataframe.groupby('n_clusters')['calinski_harabasz'].mean()

        plt.title('L\'indice de Calinski-Harabasz', fontsize=16)
        plt.plot([i for i in range(nclust_deb, 10)], s_db, marker='o')

        plt.grid(True)
        plt.xlabel('Nombre de groupes', fontsize=14)
        plt.ylabel('Indice de Calinski-Harabasz', fontsize=15)
        plt.xticks([i for i in range(nclust_deb, nclust_fin)], fontsize=14)
        plt.yticks(fontsize=15)

        plt.show()


# --------------------------------------------------------------------
# -- Affiche la géolocalisation des clusters
# --------------------------------------------------------------------

def affiche_geoloc_par_clusters(dataframe, labels):
    '''
    Affiche la géolocalisation des clusters.
    Parameters
    ----------
    dataframe : dataframe des données, obligatoire.
    labels : labels des clusters, obligatoire.
    Returns
    -------
    None.
    '''
    graph = pd.DataFrame({'Labels': labels})
    graph['Colors'] = pd.cut(graph['Labels'], bins=6,
                             labels=['blue', 'red', 'fuchsia', 'olive',
                                     'mediumpurple', 'mediumseagreen'])

    geoloc = folium.Map()
    geoloc = folium.Map(location=[dataframe['geolocation_lat'].mean(),
                                  dataframe['geolocation_lng'].mean()],
                        zoom_start=4)
    for i in range(0, 20000):
        folium.Circle([dataframe.iloc[i]['geolocation_lat'],
                       dataframe.iloc[i]['geolocation_lng']],
                      popup=graph.iloc[i]['Labels'],
                      color=graph.iloc[i]['Colors'],
                      radius=500).add_to(geoloc)

    display(geoloc)


# ###########################################################################
# -- PARTIE 11 - COMPARAISON DES ALGORITHMES
# ###########################################################################


# --------------------------------------------------------------------
# -- CALCUL DES METRIQUES pour comparaison des algorithmes
# --------------------------------------------------------------------

def calcul_comp_clustering(nom_algo, data, preds, dataframe_comparaison,
                           temps_exec):
    '''
    Calcul des métriques pour comparer les différents algorithmes de clustering.
    Parameters
    ----------
    nom_algo : nom de l'algorithme, obligatoire.
    data : données à analyser, obligatoire.
    preds : le résultat de la prédiction, obligatoire.
    dataframe_comparaison : dataframe de sauvegarde des résultats, obligatoire.
    temps_exec : temps d'exécution, obligatoire.
    Returns
    -------
    dataframe_metrique : résultat des métriques
    '''
    # Cette fonction permet de calculer le nombre de clusters le plus
    # optimal pour notre analyse : le coefficient de silhouette,
    # dispersion, indice de davies_bouldin
    silhouette = []
    davies_bouldin = []
    calin_harab = []
    algos = []
    temps = []
    nb_cluster = []

    nbcluster = len(set(preds))
    if nbcluster > 1:

        # Calcul du score de coefficient de silhouette
        silh = silhouette_score(data, preds)
        # Calcul de l'indice davies-bouldin
        db = davies_bouldin_score(data, preds)
        # Calcul de l'indice  Calinski_harabasz
        cal_har = calinski_harabasz_score(data, preds)

        silhouette.append(silh)
        davies_bouldin.append(db)
        calin_harab.append(cal_har)

    else:

        silhouette.append(0)
        davies_bouldin.append(00)
        calin_harab.append(0)

    algos.append(nom_algo)
    temps.append(temps_exec)
    nb_cluster.append(nbcluster)

    dataframe_comparaison = dataframe_comparaison.append(pd.DataFrame({
        'Algos': algos,
        'Nb_clusters': nb_cluster,
        'coef_silh': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calin_harab,
        'Durée': temps
    }), ignore_index=True)

    return dataframe_comparaison


# ###########################################################################
# -- PARTIE 12 - EVALUATION DE LA STABILITE D'INITIALISATION DES PARTITIONS
# ###########################################################################


# --------------------------------------------------------------------
# -- CALCUL DES METRIQUES pour évaluer la stabilité d'initialisation
# --------------------------------------------------------------------
def calcul_stabilite_initialisation(nom_algo, model, data, dataframe_resultat,
                                    nb_iter=5):
    '''
    CALCUL DES METRIQUES pour évaluer la stabilité d'initialisation.
    Parameters
    ----------
    nom_algo : str, nom de l'algorithme, obligatoire.
    model : instanciation du modèle dont on veut analyser la stabilité
            d'initialisation, obligatoire.
    data : données d'entréedu modèle, obligatoire.
    dataframe_resultat : dataframe de sauvegarde des résultats.
    nb_iter : nombre d'itérations (5 par défaut).
    Returns
    -------
    dataframe_resultat : dataframe de sauvegarde des résultats.
    '''
    # Initialise la liste des partitions
    partitions = []

    # Boucle sur le nombre d'itération choisi (5 par défaut)
    for i in range(nb_iter):

        # Entraînement du modèle
        model.fit(data)

        # Labels des clusters
        partitions.append(model.labels_)

    # Computing the ARI scores between partitions
    # --------------------------------------------------------

    # Initializing list of ARI scores
    ARI_scores = []

    # For each partition, except last one
    for i in range(nb_iter - 1):
        # Compute the ARI score with other partitions
        for j in range(i + 1, nb_iter):
            ARI_score = adjusted_rand_score(partitions[i], partitions[j])
            ARI_scores.append(ARI_score)

    # Compute the mean and standard deviation of ARI scores
    ARI_mean = statistics.mean(ARI_scores)
    ARI_std = statistics.stdev(ARI_scores)

    dataframe_resultat = dataframe_resultat.append(pd.DataFrame({
        'Algos': [nom_algo],
        'ARI_mean': [ARI_mean],
        'ARI_std': [ARI_std]
    }), ignore_index=True)

    return dataframe_resultat

# --------------------------------------------------------------------
# -- CALCUL DES METRIQUES pour évaluer la stabilité d'initialisation KPROTOTYPE
# --------------------------------------------------------------------


def calcul_stabilite_init_kproto(nom_algo, model, var_cat, data,
                                 dataframe_resultat, nb_iter=5):
    '''
    CALCUL DES METRIQUES pour évaluer la stabilité d'initialisation.
    Parameters
    ----------
    nom_algo : str, nom de l'algorithme, obligatoire.
    model : instanciation du modèle dont on veut analyser la stabilité
            d'initialisation, obligatoire.
    data : données d'entréedu modèle, obligatoire.
    var_cat : liste des index des variables catégorielles.
    dataframe_resultat : dataframe de sauvegarde des résultats.
    nb_iter : nombre d'itérations (5 par défaut).
    Returns
    -------
    dataframe_resultat : dataframe de sauvegarde des résultats.
    '''
    # Creating randomly initialized partitions for comparison
    # --------------------------------------------------------

    # Initializing the list of partitions
    partitions = []

    # Iterating
    for i in range(nb_iter):

        # Fitting the model
        model.fit(data, categorical=var_cat)

        # Getting the results (labels of points)
        partitions.append(model.labels_)

    # Computing the ARI scores between partitions
    # --------------------------------------------------------

    # Initializing list of ARI scores
    ARI_scores = []

    # For each partition, except last one
    for i in range(nb_iter - 1):
        # Compute the ARI score with other partitions
        for j in range(i + 1, nb_iter):
            ARI_score = adjusted_rand_score(partitions[i], partitions[j])
            ARI_scores.append(ARI_score)

    # Compute the mean and standard deviation of ARI scores
    ARI_mean = statistics.mean(ARI_scores)
    ARI_std = statistics.stdev(ARI_scores)

    # Display results
    print(
        "Evaluation of stability upon random initialization:\
        {:.1f}%  ± {:.1f}% ".format(100 * ARI_mean, 100 * ARI_std))

    dataframe_resultat = dataframe_resultat.append(pd.DataFrame({
        'Algos': [nom_algo],
        'ARI_mean': [ARI_mean],
        'ARI_std': [ARI_std]
    }), ignore_index=True)

    return dataframe_resultat

# --------------------------------------------------------------------
# -- CALCUL DES METRIQUES pour évaluer la stabilité d'initialisation KPROTOTYPE
# --------------------------------------------------------------------


def calcul_stabilite_init_gmm(nom_algo, model, data, dataframe_resultat,
                              nb_iter=5):
    '''
    CALCUL DES METRIQUES pour évaluer la stabilité d'initialisation.
    Parameters
    ----------
    nom_algo : str, nom de l'algorithme, obligatoire.
    model : instanciation du modèle dont on veut analyser la stabilité
            d'initialisation, obligatoire.
    data : données d'entréedu modèle, obligatoire.
    dataframe_resultat : dataframe de sauvegarde des résultats.
    nb_iter : nombre d'itérations (5 par défaut).
    Returns
    -------
    dataframe_resultat : dataframe de sauvegarde des résultats.
    '''
    # Creating randomly initialized partitions for comparison
    # --------------------------------------------------------

    # Initializing the list of partitions
    partitions = []

    # Iterating
    for i in range(nb_iter):

        # Fitting the model
        model.fit(data)

        # Getting the results (labels of points)
        labels = model.fit_predict(data)
        partitions.append(labels)

    # Computing the ARI scores between partitions
    # --------------------------------------------------------

    # Initializing list of ARI scores
    ARI_scores = []

    # For each partition, except last one
    for i in range(nb_iter - 1):
        # Compute the ARI score with other partitions
        for j in range(i + 1, nb_iter):
            ARI_score = adjusted_rand_score(partitions[i], partitions[j])
            ARI_scores.append(ARI_score)

    # Compute the mean and standard deviation of ARI scores
    ARI_mean = statistics.mean(ARI_scores)
    ARI_std = statistics.stdev(ARI_scores)

    # Display results
    print(
        "Evaluation of stability upon random initialization:\
        {:.1f}%  ± {:.1f}% ".format(100 * ARI_mean, 100 * ARI_std))

    dataframe_resultat = dataframe_resultat.append(pd.DataFrame({
        'Algos': [nom_algo],
        'ARI_mean': [ARI_mean],
        'ARI_std': [ARI_std]
    }), ignore_index=True)

    return dataframe_resultat

# --------------------------------------------------------------------
# -- Segmentation Kmeans à partir d'une date + métrique de stabilité
# --------------------------------------------------------------------


def segmentation_kmean_refit(dataframe, dataframe_resutat, date_ref, titre,
                             nb_clusters, seed):
    '''
    Segmentation de clientèle à partir d'une date + métrique de stabilité.
    Parameters
    ----------
    dataframe : dataframe à analyser, obligatoire.
    dataframe_resutat : dataframe de sauvegarde des scores ARI, obligatoire.
    date_ref : date de fin d'analyse de la segmentation avant la date de fin
               d'historique, string au format 'YYYY-MM-DD HH24:MI:SS'.
    titre : titre correspondant à la période de résultat pour le dataframe de
            sauvegarde des scores ARI, obligatoire.
    nb_clusters : nombre de cluster, obligatoire.
    seed : graine aléatoire pour la repoductibilité des résultats.
    Returns
    -------
    dataframe_resutat : dataframe des sauvegarde des résultats ARI
    df_km : le dataframe de segmentation Kmeans sur la période historique.
    '''

    # ------------------------------------------------------------------------
    # Préparation des dataframes de travail
    # ------------------------------------------------------------------------
    dataframe['date_dernier_achat'] = \
        pd.to_datetime(dataframe['date_dernier_achat'],
                       format='%Y-%m-%d %H:%M:%S')

    # Création des 2 tables de comparaison de la stabilité
    df_hist = \
        dataframe[dataframe['date_dernier_achat'] < date_ref]

    # Suppression colonnes
    cols_a_suppr = ['customer_unique_id', 'date_premier_achat',
                    'date_dernier_achat', 'moyen_paiement_prefere',
                    'cat_produit_prefere']
    df_hist.drop(cols_a_suppr, axis=1, inplace=True)

    # Centrage et Réduction
    # Nous mettons à l'échelle les données afin de garantir que Les unités
    #  des variables n'ont pas d'impact sur les distance
    for c in df_hist.columns:
        std_scale = StandardScaler()
        df_hist[c] = \
            std_scale.fit_transform(np.array(df_hist[c]).reshape(-1, 1))

    # ------------------------------------------------------------------------
    # Clustering Kmeans sur les 2 périodes historique et de référence
    # ------------------------------------------------------------------------
    # kmeans sur la période historique
    kmeans_hist = KMeans(n_clusters=nb_clusters, init='k-means++',
                         random_state=seed)
    kmeans_pred_hist = kmeans_hist.fit_predict(df_hist)

    # Réentrainemet de Kmeans sur la période historique
    kmeans_hist.fit(df_hist)
    kmeans_labels_refit = kmeans_hist.labels_

    # ------------------------------------------------------------------------
    # Scoring ARI de la stabilité
    # ------------------------------------------------------------------------
    # Calcul du score ARI
    ARI_kmeans = metrics.adjusted_rand_score(kmeans_pred_hist,
                                             kmeans_labels_refit)

    # Sauvegarde de l'ARI dans le tableau de résultats
    dataframe_resutat = dataframe_resutat.append(pd.DataFrame({
        'Periode': [titre],
        'ARI': [ARI_kmeans]}), ignore_index=True)

    return dataframe_resutat, df_hist


# ###########################################################################
# -- PARTIE 13 - STABILITE - EVOLUTION DES CLUSTERS DANS LE TEMPS
# ###########################################################################


# --------------------------------------------------------------------
# -- Segmentation Kmeans à partir d'une date + métrique de stabilité
# --------------------------------------------------------------------

def segmentation_kmean_periode(dataframe, dataframe_resutat, date_ref, titre,
                               nb_clusters):
    '''
    Segmentation de clientèle à partir d'une date + métrique de stabilité.
    Parameters
    ----------
    dataframe : dataframe à analyser, obligatoire.
    dataframe_resutat : dataframe de sauvegarde des scores ARI, obligatoire.
    date_ref : date de fin d'analyse de la segmentation avant la date de fin
               d'historique, string au format 'YYYY-MM-DD HH24:MI:SS'.
    titre : titre correspondant à la période de résultat pour le dataframe de
            sauvegarde des scores ARI, obligatoire.
    nb_clusters : nombre de cluster, obligatoire.
    Returns
    -------
    dataframe_resutat : dataframe des sauvegarde des résultats ARI
    df_km : le dataframe de segmentation Kmeans sur la période historique.
    '''

    # ------------------------------------------------------------------------
    # Préparation des dataframes de travail
    # ------------------------------------------------------------------------
    dataframe['date_dernier_achat'] = \
        pd.to_datetime(dataframe['date_dernier_achat'],
                       format='%Y-%m-%d %H:%M:%S')
    df_copie = dataframe.copy()
    # Création des 2 tables de comparaison de la stabilité
    df_hist = \
        df_copie[df_copie['date_dernier_achat'] < date_ref]

    # On garde les clients qui étaient dans la base de données sur la période
    # historique
    df_copie_ref = dataframe.copy()
    df_ref = df_copie_ref[df_copie_ref.customer_unique_id.isin(
        df_hist.customer_unique_id)]

    # Sélection des variables numériques
    cols_num_cat = df_ref.select_dtypes(include=[np.number]).columns.to_list()
    # Transformation en logarithme pour avoir le même poids
    col_to_log_cat = cols_num_cat[2:17]
    df_hist[col_to_log_cat] = df_hist[col_to_log_cat].apply(np.log1p, axis=1)
    df_ref[col_to_log_cat] = df_ref[col_to_log_cat].apply(np.log1p, axis=1)

    # Standardisation StandardScaler - variable transformées en log
    # -----------------------------------------------------------------------
    # Préparation des données
    X_df_hist = df_hist[cols_num_cat].values
    features_df_hist = df_hist[cols_num_cat].columns
    # Standardisation avec StandardScaler (centre, réduit et rend plus la
    # distribution plus normale)
    scaler = StandardScaler()
    X_scaled_hist = scaler.fit_transform(X_df_hist)
    # Dataframe
    X_scaled_hist = pd.DataFrame(X_scaled_hist,
                                 index=df_hist[col_to_log_cat].index,
                                 columns=features_df_hist)

    X_df_ref = df_ref[cols_num_cat].values
    features_df_ref = df_ref[cols_num_cat].columns
    # Standardisation avec StandardScaler (centre, réduit et rend plus la
    # distribution plus normale)
    scaler = StandardScaler()
    X_scaled_ref = scaler.fit_transform(X_df_ref)
    # Dataframe
    X_scaled_ref = pd.DataFrame(X_scaled_ref,
                                index=df_ref[col_to_log_cat].index,
                                columns=features_df_ref)

    # Encodage du moyen de paiement préféré
    encod_paiement = pd.get_dummies(df_hist['moyen_paiement_prefere'])
    X_scaled_hist = X_scaled_hist.join(encod_paiement)
    encod_paiement_2 = pd.get_dummies(df_ref['moyen_paiement_prefere'])
    X_scaled_ref = X_scaled_ref.join(encod_paiement_2)

    # Encodage de la catégorie préférée
    encod_cat_pref = pd.get_dummies(df_hist['cat_produit_prefere'])
    X_scaled_hist = X_scaled_hist.join(encod_cat_pref)
    encod_cat_pref_2 = pd.get_dummies(df_ref['cat_produit_prefere'])
    X_scaled_ref = X_scaled_ref.join(encod_cat_pref_2)

    # ------------------------------------------------------------------------
    # Clustering Kmeans sur les 2 périodes historique et de référence
    # ------------------------------------------------------------------------
    # Instanciation de kmeans
    kmeans = KMeans(n_clusters=nb_clusters,
                    n_init=20,
                    init='k-means++')
    # Entaînement de kmeans sur la période historique
    kmeans.fit(X_scaled_hist)
    kmeans_labels_hist = kmeans.labels_
    X_scaled_hist['Cluster'] = kmeans.labels_
    # Entaînement de sur la période de référence
    kmeans.fit(X_scaled_ref)
    kmeans_labels_ref = kmeans.labels_
    X_scaled_ref['Cluster'] = kmeans.labels_

    # ------------------------------------------------------------------------
    # Scoring ARI de la stabilité
    # ------------------------------------------------------------------------
    # Calcul du score ARI
    ARI_kmeans = metrics.adjusted_rand_score(kmeans_labels_ref,
                                             kmeans_labels_hist)

    # Sauvegarde de l'ARI dans le tableau de résultats
    dataframe_resutat = dataframe_resutat.append(pd.DataFrame({
        'Periode': [titre],
        'Date': [date_ref],
        'ARI': [ARI_kmeans]}),
        ignore_index=True)

    return dataframe_resutat, X_scaled_hist, X_scaled_ref

# --------------------------------------------------------------------
# -- Segmentation K-Prototype à partir d'une date + métrique de stabilité
# --------------------------------------------------------------------


def segmentation_kproto_periode(dataframe, dataframe_resutat, date_ref, titre,
                                nb_clusters):
    '''
    Segmentation de clientèle à partir d'une date + métrique de stabilité.
    Parameters
    ----------
    dataframe : dataframe à analyser, obligatoire.
    dataframe_resutat : dataframe de sauvegarde des scores ARI, obligatoire.
    date_ref : date de fin d'analyse de la segmentation avant la date de fin
               d'historique, string au format 'YYYY-MM-DD HH24:MI:SS'.
    titre : titre correspondant à la période de résultat pour le dataframe de
            sauvegarde des scores ARI, obligatoire.
    nb_clusters : nombre de cluster, obligatoire.
    Returns
    -------
    dataframe_resutat : dataframe des sauvegarde des résultats ARI
    df_km : le dataframe de segmentation Kmeans sur la période historique.
    '''

    # ------------------------------------------------------------------------
    # Préparation des dataframes de travail
    # ------------------------------------------------------------------------
    dataframe['date_dernier_achat'] = \
        pd.to_datetime(dataframe['date_dernier_achat'],
                       format='%Y-%m-%d %H:%M:%S')
    df_copie = dataframe.copy()
    # Création des 2 tables de comparaison de la stabilité
    df_hist = \
        df_copie[df_copie['date_dernier_achat'] < date_ref]

    # On garde les clients qui étaient dans la base de données sur la période
    # historique
    df_copie_ref = dataframe.copy()
    df_ref = df_copie_ref[df_copie_ref.customer_unique_id.isin(
        df_hist.customer_unique_id)]

    # Suppression colonnes
    cols_a_suppr = ['customer_unique_id', 'date_premier_achat',
                    'date_dernier_achat']
    df_hist.drop(cols_a_suppr, axis=1, inplace=True)
    df_ref.drop(cols_a_suppr, axis=1, inplace=True)

    # Pré-processing
    scaler = StandardScaler()
    for col in df_hist.select_dtypes(exclude='object').columns:
        if col != 'geolocation_lat':
            if col != 'geolocation_lng':
                df_hist[col] = df_hist[col].apply(np.log1p)
        df_hist[col] = scaler.fit_transform(np.array(df_hist[col])
                                              .reshape(-1, 1))
    scaler = StandardScaler()
    for col in df_ref.select_dtypes(exclude='object').columns:
        if col != 'geolocation_lat':
            if col != 'geolocation_lng':
                df_ref[col] = df_ref[col].apply(np.log1p)
        df_ref[col] = scaler.fit_transform(np.array(df_ref[col])
                                           .reshape(-1, 1))

    # Détermine les variables catégorielles par leurs index
    cols_categorical = [15, 18]

    # ------------------------------------------------------------------------
    # Clustering Kmeans sur les 2 périodes historique et de référence
    # ------------------------------------------------------------------------

    # Instanciation du modèle de clustering
    kproto = KPrototypes(n_clusters=6, init='Cao', n_jobs=-1)
    kproto.fit_predict(df_hist, categorical=cols_categorical)
    kproto_labels_hist = kproto.labels_
    df_hist['Cluster'] = kproto.labels_
    kproto.fit_predict(df_ref, categorical=cols_categorical)
    kproto_labels_ref = kproto.labels_
    df_ref['Cluster'] = kproto.labels_

    # ------------------------------------------------------------------------
    # Scoring ARI de la stabilité
    # ------------------------------------------------------------------------
    # Calcul du score ARI
    ARI_kproto = metrics.adjusted_rand_score(kproto_labels_ref,
                                             kproto_labels_hist)

    # Sauvegarde de l'ARI dans le tableau de résultats
    dataframe_resutat = dataframe_resutat.append(pd.DataFrame({
        'Periode': [titre],
        'Date': [date_ref],
        'ARI': [ARI_kproto]}),
        ignore_index=True)

    return dataframe_resutat, df_hist, df_ref
