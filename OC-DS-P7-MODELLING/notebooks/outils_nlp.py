""" Librairie personnelle pour les données textuelles NLP...
"""

#! /usr/bin/env python3
# coding: utf-8

# ====================================================================
# Outils NLP -  projet 6 Openclassrooms
# Version : 0.0.0 - CRE LR 27/05/2021
# ====================================================================
import pandas as pd
import numpy as np
import outils_data
# Traitement de text
import string
import texthero as hero
# from texthero import preprocessing
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize  # , FreqDist
from stop_words import get_stop_words
import time
# Clustering Metrics
from sklearn.metrics import davies_bouldin_score, silhouette_score, \
    adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
# Clustering
from sklearn.cluster import KMeans
# Qualité Clusters
from sklearn.metrics import confusion_matrix, classification_report, \
    accuracy_score
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
import gensim
from sklearn.manifold import TSNE
from pycaret.classification import setup, compare_models, predict_model

# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.0'

# --------------------------------------------------------------------
# -- Ajoute action, valeur dans dataframe témoin
# --------------------------------------------------------------------


def suivre_modif_texte(dataframe, action, valeur):
    '''
    Ajoute l'action et la valeur dans un dataframe de suivi des modifications
    du texte pour les différentes action effectuée.
    Parameters
    ----------
    dataframe : dataframe de suivi des modifications, obligaotire.
    action : string, obligatoire (ex : tokenisation, lemmatisation...).
    valeur : texte après avoir subi la modification de l'action, obligatoire.
    Returns
    -------
    dataframe : le dataframe de suivi des modifications avec l'action ajoutée.
    '''
    dataframe = dataframe.append(pd.DataFrame({'Action': [action],
                                               'Contenu': [valeur]}),
                                 ignore_index=True)

    return dataframe


# --------------------------------------------------------------------
# -- PRE-TRAITEMENT NLP - tokenisation, normalisation
# --------------------------------------------------------------------

def pretraiter_texte(dataframe, variable_a_traiter, var_traitee,
                     new_custom_words):
    '''
    Pré-traitment de la variable transmise (tokenisation, normalisation)
    Parameters
    ----------
    dataframe : dataframe, obligatoire.
    variable_a_traiter : variable qu'on souhaite pré-traiter, obligatoire.
    var_traitee : sauvegarde les transformations dans une nouvelle variable,
                  obligatoire.
    new_custom_words : stop words personnalisés en plus des défauts, facultatif.
    Returns
    -------
    None.
    '''
    dataframe[var_traitee] = dataframe[variable_a_traiter]

    # Tokenisation
    # ----------------------------------------------------------------------
    # Décomposition en tokens (souvent un mot mais peut être 3) par exemple)
    dataframe[var_traitee] = [word_tokenize(str(row))
                              for row in dataframe[variable_a_traiter]]
    # Transformer liste en string
    dataframe[var_traitee] = dataframe[var_traitee].apply(
        lambda x: ' '.join(x))

    # Normalisation
    # ----------------------------------------------------------------------
    # Suppression des \
    dataframe[var_traitee] = dataframe[var_traitee].replace(r'\\', ' ',
                                                            regex=True)
    # Passage en minuscules pour être insensible à la casse (stopwords en
    # minuscules)
    dataframe[var_traitee] = hero.lowercase(dataframe[var_traitee])
    # Supprimer tous les chiffres.
    dataframe[var_traitee] = hero.remove_digits(dataframe[var_traitee],
                                                only_blocks=False)
    # Supprime la ponctuation
    dataframe[var_traitee] = hero.remove_punctuation(dataframe[var_traitee])
    # Supprime les espaces blancs, retour chariot...
    dataframe[var_traitee] = hero.remove_whitespace(dataframe[var_traitee])

    # StopWords
    # ----------------------------------------------------------------------
    # Stop_words par défaut de la librairie TextHero (179 mots)
    default_stopwords = hero.stopwords.DEFAULT
    # Ajout des stops words de la librairie NLTK (174 mots)
    custom_stopwords = default_stopwords.union(get_stop_words('english'))
    # Ajout des stops words de la librairie stopwords (179 mots)
    custom_stopwords = custom_stopwords.union(stopwords.words('english'))
    # Ajout de toutes les lettres seules de a à z
    custom_stopwords = custom_stopwords.union(list(string.ascii_lowercase))
    # Stop words particulier
    if len(new_custom_words) > 1:
        custom_stopwords = custom_stopwords.union(set(new_custom_words))
    # Supprimer les stop_words anglais définis dans les 3 librairies
    dataframe[var_traitee] = hero.remove_stopwords(dataframe[var_traitee],
                                                   custom_stopwords)


# --------------------------------------------------------------------
# -- RACINISATION (STEMMING) de la variable du dataframe transmis
# --------------------------------------------------------------------

def raciniser_texte(dataframe, variable_a_raciniser, new_var):
    '''
    RACINISATION (STEMMING) de la variable du dataframe transmis.
    La racinisation(ou stemming en anglais) consiste à ne conserver que la
    racine des mots étudiés. L'idée étant de supprimer les suffixes, préfixes
    et autres des mots afin de ne conserver que leur origine.
    Mots anglais ==> stem 'porter' (pas snowball)
    Parameters
    ----------
    dataframe : dataframe, obligatoire.
    variable_a_raciniser : variable contenant du texte à raciniser.
    new_var : variable de sauvegarde de la transformation (peut être la même
              que variable_a_raciniser), obligatoire.
    Returns
    -------
    None.
    '''
    dataframe[new_var] = hero.stem(dataframe[variable_a_raciniser],
                                   stem='porter')

# --------------------------------------------------------------------
# -- LEMMATISATION de la variable du dataframe transmis
# --------------------------------------------------------------------


def lemmatiser_texte(dataframe, variable_a_lemmatiser, new_var):
    '''
    LEMMATISATION de la variable du dataframe transmis.
    Le processus de lemmatisation consiste à représenter les mots (ou lemmes)
    sous leur forme canonique pour ne conserver que le sens des mots utilisés
    dans le corpus.
    Par exemple pour un verbe, ce sera son infinitif, pour un nom, son masculin
    singulier.
    Parameters
    ----------
    dataframe : dataframe, obligatoire.
    variable_a_lemmatiser
    : variable contenant du texte à raciniser.
    new_var : variable de sauvegarde de la transformation (peut être la même
              que variable_a_raciniser), obligatoire.
    Returns
    -------
    None.
    '''
    # ---------------------------------------------------------------------
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    dataframe[new_var] = \
        [[lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
         for text in dataframe[variable_a_lemmatiser]]
    # Transformer liste en string
    dataframe[new_var] = dataframe[new_var].apply(lambda x: ' '.join(x))


# --------------------------------------------------------------------
# -- CALCUL DES METRIQUES K-Means
# --------------------------------------------------------------------

def calcul_metrics_kmeans(data, dataframe_metrique, type_donnees,
                          random_seed, ninit, maxiter):
    '''
    Calcul des métriques de KMeans en fonction de différents paramètres.
    Parameters
    ----------
    data : données, obligatoire.
    dataframe_metrique : dataframe de sauvegarde des résultats, obligatoire.
    type_donnees : string intitulé des données, obligatoire.
    random_seed : nombre aléatoire pour la reproductibilité, obligatoire.
    ninit : nombre de clusters à initialiser, obligatoire.
    maxiter : type d'initialisation : 'k-means++' ou 'random'.
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

    result_ninit = []
    result_maxiter = []

    # Hyperparametre tuning

    ninit = ninit
    maxiter = maxiter

    # Recherche des hyperparamètres
    for var_ninit in ninit:

        for var_maxiter in maxiter:

            # Top début d'exécution
            time_start = time.time()

            # Initialisation de l'algorithme
            cls = KMeans(n_clusters=7,
                         n_init=var_ninit,
                         init='k-means++',
                         max_iter=var_maxiter,
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

            result_ninit.append(var_ninit)
            result_maxiter.append(var_maxiter)

    dataframe_metrique = dataframe_metrique.append(pd.DataFrame({
        'Type_données': donnees,
        'n_init': result_ninit,
        'max_iter': result_maxiter,
        'coef_silh': silhouette,
        'dispersion': dispersion,
        'davies_bouldin': davies_bouldin,
        'Durée (s)': temps
    }), ignore_index=True)

    return dataframe_metrique

# --------------------------------------------------------------------
# -- Affiche la répartition du nombre de clients par clusters
# --------------------------------------------------------------------


def affiche_repartition_par_clusters(clusters_labels):
    '''
    Affiche la répartition par cluster
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
         'Nombre': series_client_cluster.values})
    df_visu_client_cluster['%'] = round(
        (df_visu_client_cluster['Nombre']) * 100 / nb_client, 2)
    df_visu_client_cluster = df_visu_client_cluster.sort_values(by='Clusters')
    display(df_visu_client_cluster.style.hide_index())

    # Barplot de la distribution
    sns.set_style('white')
    sns.barplot(x='Clusters', y='Nombre',
                data=df_visu_client_cluster, color='SteelBlue', ax=ax1)
    ax1.set_ylabel('Nombre)', fontsize=12)
    ax1.set_xlabel('Clusters', fontsize=12)
    ax1.set_title('Répartition par clusters', fontsize=14)
    plt.gcf().set_size_inches(6, 4)
    plt.grid(False)
    plt.show()


# --------------------------------------------------------------------
# -- KMeans - Affiche la répartition du nombre de clients par clusters
# --------------------------------------------------------------------


def calcul_metriques_clusters(dataframe, dataframe_sauv_res, type_donnee):
    '''
    Calcul de la métrique ARI, .
    Parameters
    ----------
    dataframe : dataframe, obligatoire.
    dataframe_sauv_res : dataframe de sauvegarde du résultat, obligatoire.
    type_donnee : titre pour dataframe de sauvegarde, obligatoire.
    Returns
    -------
    dataframe_sauv_res.
    '''
    valeur_reel = dataframe['CATEGORIE']
    valeur_pred = dataframe['Clusters']
    score_ari = adjusted_rand_score(valeur_reel, valeur_pred)
    score_homogeneite = homogeneity_score(valeur_reel, valeur_pred)
    score_completude = completeness_score(valeur_reel, valeur_pred)
    score_vmeasure = v_measure_score(valeur_reel, valeur_pred)

    dataframe_sauv_res = dataframe_sauv_res.append(pd.DataFrame({
        'Type_données': [type_donnee],
        'ARI': [score_ari],
        'Homogénéité': [score_homogeneite],
        'Complétude': [score_completude],
        'V-measure': [score_vmeasure]}), ignore_index=True)

    return dataframe_sauv_res


# --------------------------------------------------------------------
# -- LDA - Affiche la répartition du nombre de clients par clusters
# --------------------------------------------------------------------


def calcul_metriques_clusters_lda(dataframe, dataframe_sauv_res, type_donnee):
    '''
    Calcul de la métrique ARI,homogénéité, complétude,V-Measure .
    Parameters
    ----------
    dataframe : dataframe, obligatoire.
    dataframe_sauv_res : dataframe de sauvegarde du résultat, obligatoire.
    type_donnee : titre pour dataframe de sauvegarde, obligatoire.
    Returns
    -------
    dataframe_sauv_res.
    '''
    valeur_reel = dataframe['CATEGORIE']
    valeur_pred = dataframe['Top_topics_labels']
    score_ari = adjusted_rand_score(valeur_reel, valeur_pred)
    score_homogeneite = homogeneity_score(valeur_reel, valeur_pred)
    score_completude = completeness_score(valeur_reel, valeur_pred)
    score_vmeasure = v_measure_score(valeur_reel, valeur_pred)

    dataframe_sauv_res = dataframe_sauv_res.append(pd.DataFrame({
        'Type_données': [type_donnee],
        'ARI': [score_ari],
        'Homogénéité': [score_homogeneite],
        'Complétude': [score_completude],
        'V-measure': [score_vmeasure]}), ignore_index=True)

    return dataframe_sauv_res


# --------------------------------------------------------------------
# -- Affiche la métrique ARI du cluster
# --------------------------------------------------------------------


def calcul_metrique_ari(dataframe, dataframe_sauv_res, type_donnee):
    '''
    Calcul de la métrique ARI.
    Parameters
    ----------
    dataframe : dataframe, obligatoire.
    dataframe_sauv_res : dataframe de sauvegarde du résultat, obligatoire.
    type_donnee : titre pour dataframe de sauvegarde, obligatoire.
    Returns
    -------
    dataframe_sauv_res.
    '''
    valeur_reel = dataframe['CATEGORIE']
    valeur_pred = dataframe['Clusters']
    score_ari = adjusted_rand_score(valeur_reel, valeur_pred)

    dataframe_sauv_res = dataframe_sauv_res.append(pd.DataFrame({
        'Type_données': [type_donnee],
        'ARI': [score_ari]}), ignore_index=True)

    return dataframe_sauv_res


# --------------------------------------------------------------------
# -- Affiche une visualisation 2D des clients par clusters par t-SNE
# --------------------------------------------------------------------


def affiche_projection_par_clusters(X_projected, clusters_labels, titre):
    '''
    Affiche la projection scatterplot par clusters.
    Parameters
    ----------
    proj_tsne : projection, obligatoire.
    clusters_labels : la variable représentant les clusters.
    Returns
    -------
    None.
    '''
    plt.figure(figsize=[25, 15])

    sns.scatterplot(X_projected[:, 0], X_projected[:, 1],
                    hue=clusters_labels, s=100, alpha=1, palette='Set1')

    plt.title(titre, fontsize=40)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=34)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(False)
    plt.show()


# --------------------------------------------------------------------
# -- KMeans - Affiche HEATMAP DE LA MATRICE DE CONFUSION
# --------------------------------------------------------------------

def affiche_qualite_categorisation(dataframe, dico_traduction, titre):
    '''
    Afiche heatmap de la qualité de la catégorisation et donne l'accuracy.
    Parameters
    ----------
    dataframe : dataframe, obligatoire.
    dico_traduction : traduction des catégories de nombre à texte, obligatoire.
    titre : type de données traitées.
    Returns
    -------
    None.
    '''
    dataframe['Cluster_labels'] = dataframe['Clusters'].copy().astype('str')
    outils_data.traduire_valeurs_variable(dataframe, 'Cluster_labels',
                                          dico_traduction)

    cat_pred = dataframe['CATEGORIE']
    cat_reel = dataframe['Cluster_labels']

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(cat_reel, cat_pred),
                annot=True,
                fmt='d',
                cbar=False,
                cmap='Blues',
                yticklabels=sorted(cat_pred.unique()))
    plt.title(titre)
    plt.show()

    display(
        'Précision: {}%'.format(
            round(
                accuracy_score(
                    cat_pred,
                    cat_reel) *
                100,
                2)))
    display(Markdown(classification_report(cat_reel, cat_pred)))

# --------------------------------------------------------------------
# -- LDA - Affiche HEATMAP DE LA MATRICE DE CONFUSION
# --------------------------------------------------------------------


def affiche_qualite_categorisation_lda(dataframe, dico_traduction, titre):
    '''
    Afiche heatmap de la qualité de la catégorisation LDA et donne l'accuracy.
    Parameters
    ----------
    dataframe : dataframe, obligatoire.
    dico_traduction : traduction des catégories de nombre à texte, obligatoire.
    titre : type de données traitées.
    Returns
    -------
    None.
    '''
    dataframe['Top_topics_labels'] = dataframe['Top_topics'].copy().astype('str')
    outils_data.traduire_valeurs_variable(dataframe, 'Top_topics_labels',
                                          dico_traduction)

    cat_pred = dataframe['CATEGORIE']
    cat_reel = dataframe['Top_topics_labels']

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(cat_reel, cat_pred),
                annot=True,
                fmt='d',
                cbar=False,
                cmap='Blues',
                yticklabels=sorted(cat_pred.unique()))
    plt.title(titre)
    plt.show()

    display(
        'Précision: {}%'.format(
            round(
                accuracy_score(
                    cat_pred,
                    cat_reel) *
                100,
                2)))
    display(Markdown(classification_report(cat_reel, cat_pred)))

# --------------------------------------------------------------------
# -- Sauvegarde de la précision de la catégorisation dans un fichier excel
# --------------------------------------------------------------------


def sauvegarde_precision(dataframe, precision, titre):
    '''
    Sauvegarde de la précision de la catégorisation dans un fichier excel.
    Parameters
    ----------
    dataframe : dataframe, obligatoire.
    precision : précision de la catégorisation.
    titre : type de données traitées
    Returns
    -------
    dataframe : dataframe de sauvegarde.
    '''
    dataframe = dataframe.append(pd.DataFrame({
        'Type_données': [titre],
        'Précision': [precision]
    }), ignore_index=True)

    return dataframe

# --------------------------------------------------------------------
# -- Word2Vec - Vectorisation
# --------------------------------------------------------------------


def creer_vecteur_moyen_par_mot(data, text_dim, w2v_model):

    vect_moy = np.zeros((text_dim,), dtype='float32')
    num_words = 0.

    for word in data.split():
        if word in w2v_model.wv.vocab:
            vect_moy = np.add(vect_moy, w2v_model[word])
            num_words += 1.

    if num_words != 0.:
        vect_moy = np.divide(vect_moy, num_words)

    return vect_moy


def word2vec_vectorisation(data, text_dim, w2v_model):
    '''
    Vectorisation.
    Parameters
    ----------
    data : variable à vectoriser, obligatoire.
    text_dim : taille du vecteur, obligatoire.
    w2v_model : modèle Word2Vec entraîné, obligatoire.
    Returns
    -------
    w2v_vector : les words vectorisés.
    '''
    w2v_vector = np.zeros((data.shape[0], text_dim), dtype='float32')

    for i in range(len(data)):
        w2v_vector[i] = creer_vecteur_moyen_par_mot(
            data[i], text_dim, w2v_model)

    return w2v_vector


# --------------------------------------------------------------------
# -- TruncatedSVD - sélection du nombre de composant minimal
# --------------------------------------------------------------------

def select_n_components(var_ratio, goal_var: float) -> int:
    # Set initial variance explained so far
    total_variance = 0.0

    # Set initial number of features
    n_components = 0

    # For the explained variance of each feature:
    for explained_variance in var_ratio:

        # Add the explained variance to the total
        total_variance += explained_variance

        # Add one to the number of components
        n_components += 1

        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break

    # Return the number of components
    return n_components


# --------------------------------------------------------------------
# -- Doc2Vec - Source : https://radimrehurek.com/gensim/auto_examples/
#                       tutorials/run_doc2vec_lee.html
# --------------------------------------------------------------------


def read_corpus(dataframe, variable, tokens_only=False):

    for i, line in enumerate(dataframe[variable]):
        tokens = gensim.utils.simple_preprocess(line)

        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

# --------------------------------------------------------------------
# LDA/NMF - Visualise les mots les plus fréquents des topics
# Source :
# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_
# extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics
# -extraction-with-nmf-lda-py
# --------------------------------------------------------------------


def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(3, 3, figsize=(15, 30), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

# --------------------------------------------------------------------
# -- Pycaret - Affiche la répartition du nombre de clients par clusters
# --------------------------------------------------------------------


def affiche_repartition_clusters(clusters_labels):
    '''
    Affiche la répartition par cluster
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
         'Nombre': series_client_cluster.values})
    df_visu_client_cluster['%'] = round(
        (df_visu_client_cluster['Nombre']) * 100 / nb_client, 2)
    df_visu_client_cluster = df_visu_client_cluster.sort_values(by='Clusters')
    display(df_visu_client_cluster.style.hide_index())

    # Barplot de la distribution
    sns.set_style('white')
    sns.barplot(x='Clusters', y='Nombre',
                data=df_visu_client_cluster, color='SteelBlue', ax=ax1)
    ax1.set_ylabel('Nombre)', fontsize=12)
    ax1.set_xlabel('Clusters', fontsize=12)
    ax1.set_title('Répartition par clusters', fontsize=14)
    ax1.tick_params(axis='x', labelrotation=90)
    plt.gcf().set_size_inches(6, 4)
    plt.grid(False)
    plt.show()

# --------------------------------------------------------------------
# -- Pycaret - Affiche HEATMAP DE LA MATRICE DE CONFUSION
# --------------------------------------------------------------------


def affiche_qualite_classification(dataframe, titre):
    '''
    Afiche heatmap de la qualité de la catégorisation et donne l'accuracy.
    Parameters
    ----------
    dataframe : dataframe, obligatoire.
    titre : type de données traitées.
    Returns
    -------
    None.
    '''
    cat_pred = dataframe['Label']
    cat_reel = dataframe['CATEGORIE']

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(cat_reel, cat_pred),
                annot=True,
                fmt='d',
                cbar=False,
                cmap='Blues',
                yticklabels=sorted(cat_pred.unique()))
    plt.title(titre)
    plt.show()

    display(
        'Précision: {}%'.format(
            round(
                accuracy_score(
                    cat_pred,
                    cat_reel) *
                100,
                2)))
    display(Markdown(classification_report(cat_reel, cat_pred)))


# --------------------------------------------------------------------
# -- Pycaret - Affiche HEATMAP DE LA MATRICE DE CONFUSION
# --------------------------------------------------------------------

def classifier_pycaret(data_cat, vector, titre):

    # Réduction de dimension
    tsne = TSNE(verbose=1, perplexity=50, n_iter=5000)
    X_proj_tsne = tsne.fit_transform(vector)
    # Dataframe pour clustering
    df_class = pd.DataFrame({'VAR1': X_proj_tsne[:, 0],
                             'VAR2': X_proj_tsne[:, 1],
                             'CATEGORIE': data_cat})

    # Classification PyCaret
    # Initialisation
    setup(df_class,
          target='CATEGORIE',
          session_id=21,
          log_experiment=True,
          experiment_name=titre,
          silent=True)

    # Lancement de pycaret
    best_model = compare_models()

    # Prédictions
    df_result = predict_model(best_model)

    # Labels
    labels = df_result['Label']

    # Répartition des clusters
    affiche_repartition_clusters(labels)

    # Heatmap de matrice de confusion
    affiche_qualite_classification(df_result, titre)

# --------------------------------------------------------------------
# -- Pycaret - Affiche HEATMAP DE LA MATRICE DE CONFUSION
# --------------------------------------------------------------------


def classifier_pycaret_pred(data_cat, vector, titre):

    # Réduction de dimension
    tsne = TSNE(verbose=1, perplexity=50, n_iter=5000)
    X_proj_tsne = tsne.fit_transform(vector)
    # Dataframe pour clustering
    df_class = pd.DataFrame({'VAR1': X_proj_tsne[:, 0],
                             'VAR2': X_proj_tsne[:, 1],
                             'CATEGORIE': data_cat})

    # Classification PyCaret
    # Initialisation
    setup(df_class,
          target='CATEGORIE',
          session_id=21,
          log_experiment=True,
          experiment_name=titre,
          silent=True)

    # Lancement de pycaret
    best_model = compare_models()

    # Prédictions
    df_result = predict_model(best_model)

    # Labels
    labels = df_result['Label']

    # Répartition des clusters
    affiche_repartition_clusters(labels)

    # Heatmap de matrice de confusion
    affiche_qualite_classification(df_result, titre)

    return df_result


# --------------------------------------------------------------------
# -- Pycaret - Affiche HEATMAP DE LA MATRICE DE CONFUSION
# --------------------------------------------------------------------

def sauvegarder_accuracy_pycaret(dataframe, score_train, score_test, titre):
    '''

    Parameters
    ----------
    dataframe : TYPE
        DESCRIPTION.
    score_train : TYPE
        DESCRIPTION.
    score_test : TYPE
        DESCRIPTION.
    titre : TYPE
        DESCRIPTION.

    Returns
    -------
    dataframe_metrics : TYPE
        DESCRIPTION.
    '''
    dataframe_metrics = dataframe.append(pd.DataFrame({
        'Type_données': [titre],
        'Accuracy_train': [score_train],
        'Accuracy_test': [score_test]
    }), ignore_index=True)

    dataframe_metrics.to_csv('data_accuracy_classification.csv')

    return dataframe_metrics

# --------------------------------------------------------------------
# -- Pycaret - Affiche HEATMAP DE LA MATRICE DE CONFUSION
# --------------------------------------------------------------------


def sauvegarder_accuracy_we_pycaret(dataframe, score_train, score_test, titre):
    '''

    Parameters
    ----------
    dataframe : TYPE
        DESCRIPTION.
    score_train : TYPE
        DESCRIPTION.
    score_test : TYPE
        DESCRIPTION.
    titre : TYPE
        DESCRIPTION.

    Returns
    -------
    dataframe_metrics : TYPE
        DESCRIPTION.
    '''
    dataframe_metrics = dataframe.append(pd.DataFrame({
        'Type_données': [titre],
        'Accuracy_train': [score_train],
        'Accuracy_test': [score_test]
    }), ignore_index=True)

    dataframe_metrics.to_csv('data_accuracy_classification_we.csv')

    return dataframe_metrics
