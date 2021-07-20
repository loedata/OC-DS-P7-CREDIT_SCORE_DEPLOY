""" Librairie personnelle pour les données IMAGES...
"""

#! /usr/bin/env python3
# coding: utf-8

# ====================================================================
# Outils NLP -  projet 6 Openclassrooms
# Version : 0.0.0 - CRE LR 21/06/2021
# ====================================================================
# import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
# from sklearn.metrics import davies_bouldin_score, silhouette_score
# Clustering
from sklearn.cluster import KMeans
from PIL import Image, ImageOps  # , ImageFilter
import cv2
from scipy.spatial import distance
from IPython.display import display
from PIL import Image as Image_PIL

# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.0'

# --------------------------------------------------------------------
# -- Ajoute action, valeur dans dataframe témoin
# --------------------------------------------------------------------


def afficher_image_histopixel(image, titre):
    '''
    Afficher côte à côte l'image et l'histogramme de répartiton des pixels.
    Parameters
    ----------
    image : image à afficher, obligatoire.
    Returns
    -------
    None.
    '''
    plt.figure(figsize=(40, 10))
    plt.subplot(131)
    plt.title(titre, fontsize=30)
    plt.imshow(image, cmap='gray')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.subplot(132)
    plt.title('Histogramme de répartition des pixels', fontsize=30)
    hist, bins = np.histogram(np.array(image).flatten(), bins=256)
    plt.bar(range(len(hist[0:255])), hist[0:255])
    plt.xlabel('Niveau de gris', fontsize=30)
    plt.ylabel('Nombre de pixels', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.subplot(133)
    plt.title('Histogramme cumulé des pixels', fontsize=30)
    plt.hist(np.array(image).flatten(), bins=range(256), cumulative=True)
    plt.xlabel('Niveau de gris', fontsize=24)
    plt.ylabel('Fréquence cumulée de pixels', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.show()


# --------------------------------------------------------------------
# -- PRÉ-TRAITEMENT DE L'IMAGE - POUR UNE IMAGE
# --------------------------------------------------------------------


def preprocess_image(image):
    '''
    Suite aux différents tests réalisés avec les librairies PILS, OpenCv et
    Scipy, le pré-traitement suivant sera appliqué sur toutes les images :
    - Correction de l'exposition (étirement d'histogramme) avec PILS
      (autocontrast).
    - Correction du contraste (égalisation d'histogramme) avec OpenCV (CLAHE
      (Contrast Limited Adaptive Histogram Equalization).
    - Réduction du bruit avec l'algorithme Non-local Means Denoising d'OpenCV.
    - Conversion en niveau de gris de l'image (ORB, SIFT...).
    - Réduction de dimension avec OpenCV (resize et interpolation INTER_AREA).
    Parameters
    ----------
    image : image localisée dans un répertoire, obligatoire.
    Returns
    -------
    None
    '''
    # Variables locales
    dim = (224, 224)

    # Nom de l'image
    file_dir = os.path.split(image)

    # Chargement de l'image originale
    img = Image.open(image)

    # Correction de l'exposition PILS (étirement de l'histogramme)
    img = ImageOps.autocontrast(img, 1)

    # Conversion en niveau de gris de l'image
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

    # Correction du contraste OpenCV CLAHE (égalisation de l'histogramme)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    img = clahe.apply(img)

    # Réduction du bruit avec l'algorithme Non-local Means Denoising d'OpenCV
    img = cv2.fastNlMeansDenoising(img, None, 5, 7, 21)

    # Redimensionnement en 224 * 224
    img = cv2.resize(np.array(img), dim, interpolation=cv2.INTER_AREA)

    # Sauvegarde de l'image dans le répertoire data/Images_process
    cv2.imwrite('data/Images_process/' + file_dir[1], img)

    return 'data/Images_process/' + file_dir[1]


def process_image_opencv(image):
    '''
    Correction des images uniquement avec la librairie OpenCV.
    Parameters
    ----------
    image : image localisée dans un répertoire, obligatoire.
    Returns
    -------
    None
    '''
    file_dir = os.path.split(image)
    img = np.array(Image_PIL.open(image))

    # Conversion en niveau de gris de l'image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Suppression du bruit avec un kernel
    img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)

    # Egalisation de l'histogramme
    img = cv2.equalizeHist(img)

    # redimension
    dim = (224, 224)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # save image
    cv2.imwrite('data/Images_process2/' + file_dir[1], img)

    return 'data/Images_process2/' + file_dir[1]


# --------------------------------------------------------------------
# -- AFFICHER les 16 premiers VISUAL WORDS
# --------------------------------------------------------------------


def afficher_visual_words(image, keypoints):
    '''
    Afficher les 16 premiers Visual Words d'une image.
    Parameters
    ----------
    image : image, obligatoire.
    keypoints : les visual words de l'image, obligatoire.
    Returns
    -------
    None.
    '''
    plt.figure(figsize=(10, 10))
    plt.title('SIFT Visual Words des 16 premiers descripteurs')
    for i, kp in enumerate(keypoints[0:16]):
        # Get the coordinates of the center and size
        plt.subplot(4, 4, i + 1)
        x_center = kp.pt[0]
        y_center = kp.pt[1]
        size = kp.size

        # Set the border limits
        left = np.ceil(x_center - size / 2)
        upper = np.ceil(y_center + size / 2)
        right = np.ceil(x_center + size / 2)
        lower = np.ceil(y_center - size / 2)

        # Crop the image and show the parts
        cropped_np = np.array(image)[
            int(lower):int(upper), int(left):int(right)]
        plt.imshow(cropped_np)
    plt.show()

# --------------------------------------------------------------------
# -- ALGORITHME SIFT
# --------------------------------------------------------------------

 # Creates descriptors using sift
# Takes one parameter that is images dictionary
# Return an array whose first index holds the decriptor_list without an order
# And the second index holds the sift_vectors dictionary which holds the
# descriptors but this is seperated class by class


def sift_extraire_features(images):
    '''
    Extraire les descripteurs et keypoints avec SIFT.
    Parameters
    ----------
    images : les images dont on veut extraire les descripteurs et centres
             d'intérêt, obligatoire.
    Returns
    -------
    list des descripteurs et des vecteurs SIFT.
    '''
    sift_vectors = {}
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    for key, value in images.items():
        features = []
        kp, des = sift.detectAndCompute(value, None)
        descriptor_list.extend(des)
        # in case no descriptor
        des = [np.zeros((128,))] if des is None else des
        features.append(des)
        sift_vectors[key] = features
    return [descriptor_list, sift_vectors]


# --------------------------------------------------------------------
# -- ALGORITHME ORB
# --------------------------------------------------------------------


def orb_extraire_features(images):
    '''
    Extraire les descripteurs et keypoints avec ORB.
    Parameters
    ----------
    images : les images dont on veut extraire les descripteurs et centres
             d'intérêt, obligatoire.
    Returns
    -------
    list des descripteurs et des vecteurs ORB.
    '''
    orb_vectors = {}
    descriptor_list = []
    orb = cv2.ORB_create(nfeatures=1500)
    for key, value in images.items():
        # display(key)
        features = []
        kp, des = orb.detectAndCompute(value, None)
        # in case no descriptor
        des = [np.zeros((128,))] if des is None else des
        descriptor_list.extend(np.float32(des))
        features.append(des)
        orb_vectors[key] = features
    return [descriptor_list, orb_vectors]

# --------------------------------------------------------------------
# -- ALGORITHME ORB
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# -- ORB - Chargement des images et extraction des features
# --------------------------------------------------------------------


def orb_charger_img_extraire_vw(rep_images):
    '''
    ORB Charger les images et extraire les features visual words.
    Parameters
    ----------
    rep_images : répertoire de localisation des images dont on veut extraire
                 les features (obligatoire)
    Returns
    -------
    descriptors : liste des descripteurs.
    lien_img_des : lien entre image et descripteur.
    '''
    # Liste des descripteurs
    descriptor_list = []

    # Detecteur de feature ORB
    orb = cv2.ORB_create(nfeatures=1500)

    # Répertoire des images
    images = rep_images

    # Conserver le lien entre l'image et le descripteur
    lien_img_des = []

    # Pour toutes les images
    for i in range(len(images)):
        im = images[i]
        img = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(img, None)

        descriptor_list.append(des)

        # Lien image - descripteur
        for j in range(len(des)):
            lien_img_des.append(i)

        descriptors = np.array(descriptor_list[0])
        for descriptor in descriptor_list[1:]:
            descriptors = np.vstack((descriptors, descriptor))

    descriptors = np.float32(descriptors)

    return descriptors, lien_img_des

# --------------------------------------------------------------------
# -- ORB - Extraction des BOVW
# --------------------------------------------------------------------


def orb_constituter_bovw(dataframe, rep_images, dest_var_bovw, orb_labels,
                         orb_lien_img_des):
    '''
    ORB Charger les images et extraire les features visual words.
    Parameters
    ----------
    dataframe : dataframe de travail, obligatoire.
    rep_images : répertoire de localisation des images dont on veut extraire
                 les features (obligatoire)
    dest_var_bovw : variable des sotckages des BOVW,obligatoire.
    orb_labels : labels.
    orb_lien_img_des : Lien de destination des images.
    Returns
    -------
    None.
    '''
    # Constitution des histogrammes à partir des labels KMeans
    size = orb_labels.shape[0] * orb_labels.shape[1]
    data_images = []
    images = dataframe[rep_images]
    for i in range(len(images)):
        # create a numpy to hold the histogram for each image
        data_images.insert(i, np.zeros((1000, 1)))

    # Sauvegarde des BOVW de chaque image
    dataframe[dest_var_bovw] = ""
    for i in range(size):
        label = orb_labels[i]
        # Get this descriptors image id
        image_id = orb_lien_img_des[i]
        # data_images is a list of the same size as the number of images
        images_data = data_images[image_id]
        # data is a numpy array of size (dictionary_size, 1) filled with zeros
        images_data[label] += 1
        dataframe[dest_var_bovw][image_id] = images_data.flatten()


# --------------------------------------------------------------------
# -- CALCUL DES METRIQUES K-Means - Recherche paramètre k
# --------------------------------------------------------------------

def calcul_metrics_kmeans(liste_visual_words, dataframe_metrique, type_algo,
                          random_seed, nb_cluster):
    '''
    Calcul des métriques de KMeans en fonction de différents paramètres.
    Parameters
    ----------
    liste_visual_words : données, obligatoire.
    dataframe_metrique : dataframe de sauvegarde des résultats, obligatoire.
    type_algo : string type d'algorithme SIFT, ORB..., obligatoire.
    random_seed : nombre aléatoire pour la reproductibilité, obligatoire.
    nb_cluster : liste du nombre de clusters à
    Returns
    -------
    dataframe_metrique : résultat des métriques
    '''
    # Cette fonction permet de calculer le nombre de clusters le plus
    # optimal pour notre analyse : la dispersion
    # silhouette = []
    dispersion = []
    # davies_bouldin = []
    donnees = []
    temps = []

    result_k = []

    # Hyperparametre tuning

    # Recherche des hyperparamètres
    for var_k in nb_cluster:

        # Top début d'exécution
        time_start = time.time()

        # Initialisation de l'algorithme
        cls = KMeans(n_clusters=var_k,
                     init='k-means++',
                     random_state=random_seed)

        # Entraînement de l'algorithme
        cls.fit(liste_visual_words)

        # Prédictions
        # preds = cls.predict(liste_visual_words)

        # Top fin d'exécution
        time_end = time.time()

        # Calcul du score de coefficient de silhouette
        # silh = silhouette_score(liste_visual_words, preds)
        # Calcul la dispersion
        disp = cls.inertia_
        # Calcul de l'indice davies-bouldin
        # db = davies_bouldin_score(liste_visual_words, preds)
        # Durée d'exécution
        time_execution = time_end - time_start
        display('Paramètre ' + str(var_k) + ' terminé')

        # silhouette.append(silh)
        dispersion.append(disp)
        # davies_bouldin.append(db)
        donnees.append(type_algo)
        temps.append(time_execution)

        result_k.append(var_k)

    dataframe_metrique = dataframe_metrique.append(pd.DataFrame({
        'Type_données': donnees,
        'Param_k': result_k,
        # 'coef_silh': silhouette,
        'dispersion': dispersion,
        # 'davies_bouldin': davies_bouldin,
        'Durée (s)': temps
    }), ignore_index=True)

    return dataframe_metrique

# --------------------------------------------------------------------
# -- CREATION DE L'HISTOGRAMME POUR CHAQUE IMAGE
# Source
# https://github.com/AybukeYALCINER/gabor_sift_bovw/blob/
# 12d54f34226ec448f2e6f0fbc7c9a10052527368/assignment1.py#L157
# --------------------------------------------------------------------

# Takes 2 parameters. The first one is a dictionary that holds the descriptors that are separated class by class
# And the second parameter is an array that holds the central points (visual words) of the k means clustering
# Returns a dictionary that holds the histograms for each images that are
# separated class by class.


def image_class(all_bovw, centers):
    dict_feature = {}
    for key, value in all_bovw.items():
        # display(key)
        category = []
        for img in value:
            # display(img)
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature


# Find the index of the closest central point to the each sift descriptor.
# Takes 2 parameters the first one is a sift descriptor and the second one is the array of central points in k means
# Returns the index of the closest central point.
def find_index(image, center):
    count = 0
    ind = 0
    for i in range(len(center)):
        if(i == 0):
            count = distance.euclidean(image, center[i])
            #count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i])
            #dist = L1_dist(image, center[i])
            if(dist < count):
                ind = i
                count = dist
    return ind


# --------------------------------------------------------------------
# -- PARTIE CNN - Convolution Neural Network
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# -- Constituer le vecteur des features
# --------------------------------------------------------------------

def constituer_dataframe_vectors(dataframe, variable):
    '''
    Transformer les np.array des histogrammes en autant de colonnes pour
    être utilisable par TSNE et le clustering.
    Parameters
    ----------
    dataframe : dataframe des histogrammes des images.
    variable : nom de la variable contenant les histogrammes à extraire.
    Returns
    -------
    dataframe_vecteurs : le dataframe avec chacune des BOVW de chaque umage
    dans une colonne.
    '''
    vectors = np.column_stack(dataframe[variable].values.tolist())
    dataframe_vecteurs = pd.DataFrame(vectors).T

    return dataframe_vecteurs

# --------------------------------------------------------------------
# -- Pycaret - Affiche HEATMAP DE LA MATRICE DE CONFUSION
# --------------------------------------------------------------------


def sauvegarder_accuracy_class_sup_pycaret(dataframe, score_train, score_test,
                                           titre):
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

    dataframe_metrics.to_csv('IMAGES_data_scores_class_sup.csv')

    return dataframe_metrics

# --------------------------------------------------------------------
# -- Redimensionner une image en 224*224 sans
# --------------------------------------------------------------------


def cnn_redimensionner_image(image):
    '''
    Image à redimensionner en 224*224 en conservant les proportions.
    Parameters
    ----------
    image : image à sauvegarder, obligatoire.
    Returns
    -------
    Répertoire de localisation de l'image redimensionnée.
    '''
    size = 224, 224
    # On charge l'image d'origine
    im = Image.open(image)
    file_dir = os.path.split(image)
    # L'un des côté de l'image fait 224, on garde le ratio original
    # (pas de déformation)
    im.thumbnail(size, Image.ANTIALIAS)
    # On enregistre dans un nouveau dossier l'image redimensionnée.
    im.save('data/Images_redim/' + file_dir[1])

    # Centrage
    im = Image.open('data/Images_redim/' + file_dir[1])
    width, height = im.size

    if height > width:
        img = Image.new('RGB', (224, 224), (255, 255, 255))  # white
        position_larg = int((height - width) / 2)
        img.paste(im, box=(position_larg, 0))
        img.save('data/Images_redim/' + file_dir[1])

    elif width > height:
        img = Image.new('RGB', (224, 224), (255, 255, 255))  # white
        position_haut = int((width - height) / 2)
        img.paste(im, box=(0, position_haut))
        img.save("data/Images_redim/" + file_dir[1])

    return 'data/Images_redim/' + file_dir[1]


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

    dataframe_metrics.to_csv('data_accuracy_classification_images.csv')

    return dataframe_metrics
