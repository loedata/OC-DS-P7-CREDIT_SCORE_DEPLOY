""" Librairie contenant les fonctions de pré-processing.
"""

#! /usr/bin/env python3
# coding: utf-8

# ====================================================================
# Outils Fonctions PRE PROCESSING -  projet 7 Openclassrooms
# Version : 0.0.0 - CRE LR 16/07/2021
# ====================================================================
# from IPython.core.display import display
# from datetime import datetime
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from boruta import BorutaPy
from sklearn.utils import check_random_state

# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.0'


# --------------------------------------------------------------------
# -- AMELIORATION DE L'USAGE DE LA MEMOIRE DES OBJETS
# --------------------------------------------------------------------

def reduce_mem_usage(data, verbose=True):
    # source: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    '''
    This function is used to reduce the memory usage by converting the datatypes of a pandas
    DataFrame withing required limits.
    '''

    start_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('-' * 79)
        print('Memory usage du dataframe: {:.2f} MB'.format(start_mem))

    for col in data.columns:
        col_type = data[col].dtype

        #  Float et int
        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(
                        np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)
            else:
                if c_min > np.finfo(
                        np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

        # # Boolean : pas à faire car pour machine learning il faut des int 0/1
        # et pas False/True
        # if list(data[col].unique()) == [0, 1] or list(data[col].unique()) == [1, 0]:
        #     data[col] = data[col].astype(bool)

    end_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage après optimization: {:.2f} MB'.format(end_mem))
        print('Diminution de {:.1f}%'.format(
            100 * (start_mem - end_mem) / start_mem))
        print('-' * 79)

    return data


def convert_types(dataframe, print_info=False):

    original_memory = dataframe.memory_usage().sum()

    # Iterate through each column
    for c in dataframe:

        # Convert ids and booleans to integers
        if ('SK_ID' in c):
            dataframe[c] = dataframe[c].fillna(0).astype(np.int32)

        # Convert objects to category
        elif (dataframe[c].dtype == 'object') and (dataframe[c].nunique() < dataframe.shape[0]):
            dataframe[c] = dataframe[c].astype('category')

        # Booleans mapped to integers
        elif list(dataframe[c].unique()) == [1, 0]:
            dataframe[c] = dataframe[c].astype(bool)

        # Float64 to float32
        elif dataframe[c].dtype == float:
            dataframe[c] = dataframe[c].astype(np.float32)

        # Int64 to int32
        elif dataframe[c].dtype == int:
            dataframe[c] = dataframe[c].astype(np.int32)

    new_memory = dataframe.memory_usage().sum()

    if print_info:
        print(
            f'Memory Usage à l\'origine : {round(original_memory / 1e9, 2)} Gb.')
        print(
            f'Memory Usage après modification des types: {round(new_memory / 1e9, 2)} Gb.')

    return dataframe

# --------------------------------------------------------------------
# -- FEATURE ENGINEERING : création de nouvelles variables
# --------------------------------------------------------------------


def feature_engineering_application(data):
    '''
    FEATURE ENGINEERING : création de nouvelles variables.
    Extrait de : https://github.com/rishabhrao1997/Home-Credit-Default-Risk
    Parameters
    ----------
    data : dataframe pour ajout de nouvelles variables, obligatoire.
    Returns
    -------
    None.
    '''

    # -----------------------------------------------------------------------
    # Variables de revenu, de rente et de crédit :  ratio / différence
    # -----------------------------------------------------------------------
    # Ratio : Montant du crédit du prêt / Revenu du demandeur
    data['CREDIT_INCOME_RATIO'] = data['AMT_CREDIT'] / \
        (data['AMT_INCOME_TOTAL'] + 0.00001)
    # Ratio : Montant du crédit du prêt / Annuité de prêt
    data['CREDIT_ANNUITY_RATIO'] = data['AMT_CREDIT'] / \
        (data['AMT_ANNUITY'] + 0.00001)
    # Ratio : Annuité de prêt / Revenu du demandeur
    data['ANNUITY_INCOME_RATIO'] = data['AMT_ANNUITY'] / \
        (data['AMT_INCOME_TOTAL'] + 0.00001)
    # Différence : Revenu du demandeur - Annuité de prêt
    data['INCOME_ANNUITY_DIFF'] = data['AMT_INCOME_TOTAL'] - \
        data['AMT_ANNUITY']
    # Ratio : Montant du crédit du prêt / prix des biens pour lesquels le prêt est accordé
    # Crédit est supérieur au prix des biens ?
    data['CREDIT_GOODS_RATIO'] = data['AMT_CREDIT'] / \
        (data['AMT_GOODS_PRICE'] + 0.00001)
    # Différence : Revenu du demandeur - prix des biens pour lesquels le prêt
    # est accordé
    data['INCOME_GOODS_DIFF'] = data['AMT_INCOME_TOTAL'] / \
        data['AMT_GOODS_PRICE']
    # Ratio : Annuité de prêt / Âge du demandeur au moment de la demande
    data['INCOME_AGE_RATIO'] = data['AMT_INCOME_TOTAL'] / (
        data['DAYS_BIRTH'] + 0.00001)
    # Ratio : Montant du crédit du prêt / Âge du demandeur au moment de la
    # demande
    data['CREDIT_AGE_RATIO'] = data['AMT_CREDIT'] / (
        data['DAYS_BIRTH'] + 0.00001)
    # Ratio : Revenu du demandeur / Score normalisé de la source de données
    # externe 3
    data['INCOME_EXT_RATIO'] = data['AMT_INCOME_TOTAL'] / \
        (data['EXT_SOURCE_3'] + 0.00001)
    # Ratio : Montant du crédit du prêt / Score normalisé de la source de
    # données externe
    data['CREDIT_EXT_RATIO'] = data['AMT_CREDIT'] / \
        (data['EXT_SOURCE_3'] + 0.00001)
    # Multiplication : Revenu du demandeur
    #                  * heure à laquelle le demandeur à fait sa demande de prêt
    data['HOUR_PROCESS_CREDIT_MUL'] = data['AMT_CREDIT'] * \
        data['HOUR_APPR_PROCESS_START']

    # -----------------------------------------------------------------------
    # Variables sur l'âge
    # -----------------------------------------------------------------------
    # YEARS_BIRTH - Âge du demandeur au moment de la demande DAYS_BIRTH en
    # années
    data['YEARS_BIRTH'] = data['DAYS_BIRTH'] * -1 / 365
    # Différence : Âge du demandeur - Ancienneté dans l'emploi à date demande
    data['AGE_EMPLOYED_DIFF'] = data['DAYS_BIRTH'] - data['DAYS_EMPLOYED']
    # Ratio : Ancienneté dans l'emploi à date demande / Âge du demandeur
    data['EMPLOYED_AGE_RATIO'] = data['DAYS_EMPLOYED'] / \
        (data['DAYS_BIRTH'] + 0.00001)
    # Ratio : nombre de jours avant la demande où le demandeur a changé de téléphone \
    #         äge du client
    data['LAST_PHONE_BIRTH_RATIO'] = data[
        'DAYS_LAST_PHONE_CHANGE'] / (data['DAYS_BIRTH'] + 0.00001)
    # Ratio : nombre de jours avant la demande où le demandeur a changé de téléphone \
    #         ancienneté dans l'emploi
    data['LAST_PHONE_EMPLOYED_RATIO'] = data[
        'DAYS_LAST_PHONE_CHANGE'] / (data['DAYS_EMPLOYED'] + 0.00001)

    # -----------------------------------------------------------------------
    # Variables sur la voiture
    # -----------------------------------------------------------------------
    # Différence : Âge de la voiture du demandeur -  Ancienneté dans l'emploi
    # à date demande
    data['CAR_EMPLOYED_DIFF'] = data['OWN_CAR_AGE'] - data['DAYS_EMPLOYED']
    # Ratio : Âge de la voiture du demandeur / Ancienneté dans l'emploi à date
    # demande
    data['CAR_EMPLOYED_RATIO'] = data['OWN_CAR_AGE'] / \
        (data['DAYS_EMPLOYED'] + 0.00001)
    # Différence : Âge du demandeur - Âge de la voiture du demandeur
    data['CAR_AGE_DIFF'] = data['DAYS_BIRTH'] - data['OWN_CAR_AGE']
    # Ratio : Âge de la voiture du demandeur / Âge du demandeur
    data['CAR_AGE_RATIO'] = data['OWN_CAR_AGE'] / \
        (data['DAYS_BIRTH'] + 0.00001)

    # -----------------------------------------------------------------------
    # Variables sur les contacts
    # -----------------------------------------------------------------------
    # Somme : téléphone portable? + téléphone professionnel? + téléphone
    #         professionnel fixe? + téléphone portable joignable? +
    #         adresse de messagerie électronique?
    data['FLAG_CONTACTS_SUM'] = data['FLAG_MOBIL'] + data['FLAG_EMP_PHONE'] + \
        data['FLAG_WORK_PHONE'] + data['FLAG_CONT_MOBILE'] + \
        data['FLAG_PHONE'] + data['FLAG_EMAIL']

    # -----------------------------------------------------------------------
    # Variables sur les membres de la famille
    # -----------------------------------------------------------------------
    # Différence : membres de la famille - enfants (adultes)
    data['CNT_NON_CHILDREN'] = data['CNT_FAM_MEMBERS'] - data['CNT_CHILDREN']
    # Ratio : nombre d'enfants / Revenu du demandeur
    data['CHILDREN_INCOME_RATIO'] = data['CNT_CHILDREN'] / \
        (data['AMT_INCOME_TOTAL'] + 0.00001)
    # Ratio : Revenu du demandeur / membres de la famille : revenu par tête
    data['PER_CAPITA_INCOME'] = data['AMT_INCOME_TOTAL'] / \
        (data['CNT_FAM_MEMBERS'] + 1)

    # -----------------------------------------------------------------------
    # Variables sur la région
    # -----------------------------------------------------------------------
    # Moyenne : moyenne de notes de la région/ville où vit le client * revenu
    # du demandeur
    data['REGIONS_INCOME_MOY'] = (data['REGION_RATING_CLIENT'] +
                                  data['REGION_RATING_CLIENT_W_CITY']) * data['AMT_INCOME_TOTAL'] / 2
    # Max : meilleure note de la région/ville où vit le client
    data['REGION_RATING_MAX'] = [max(ele1, ele2) for ele1, ele2 in zip(
        data['REGION_RATING_CLIENT'], data['REGION_RATING_CLIENT_W_CITY'])]
    # Min : plus faible note de la région/ville où vit le client
    data['REGION_RATING_MAX'] = [min(ele1, ele2) for ele1, ele2 in zip(
        data['REGION_RATING_CLIENT'], data['REGION_RATING_CLIENT_W_CITY'])]
    # Moyenne : des notes de la région et de la ville où vit le client
    data['REGION_RATING_MEAN'] = (
        data['REGION_RATING_CLIENT'] + data['REGION_RATING_CLIENT_W_CITY']) / 2
    # Multipication : note de la région/ note de la ville où vit le client
    data['REGION_RATING_MUL'] = data['REGION_RATING_CLIENT'] * \
        data['REGION_RATING_CLIENT_W_CITY']
    # Somme : des indicateurs  :
    # Indicateur si l'adresse permanente du client ne correspond pas à l'adresse de contact (1=différent ou 0=identique - au niveau de la région)
    # Indicateur si l'adresse permanente du client ne correspond pas à l'adresse professionnelle (1=différent ou 0=identique - au niveau de la région)
    # Indicateur si l'adresse de contact du client ne correspond pas à l'adresse de travail (1=différent ou 0=identique - au niveau de la région).
    # Indicateur si l'adresse permanente du client ne correspond pas à l'adresse de contact (1=différent ou 0=identique - au niveau de la ville)
    # Indicateur si l'adresse permanente du client ne correspond pas à l'adresse professionnelle (1=différent ou 0=même - au niveau de la ville).
    # Indicateur si l'adresse de contact du client ne correspond pas à
    # l'adresse de travail (1=différent ou 0=identique - au niveau de la
    # ville).
    data['FLAG_REGIONS_SUM'] = data['REG_REGION_NOT_LIVE_REGION'] + \
        data['REG_REGION_NOT_WORK_REGION'] + \
        data['LIVE_REGION_NOT_WORK_REGION'] + \
        data['REG_CITY_NOT_LIVE_CITY'] + \
        data['REG_CITY_NOT_WORK_CITY'] + \
        data['LIVE_CITY_NOT_WORK_CITY']

    # -----------------------------------------------------------------------
    # Variables sur les sources externes : sum, min, multiplication, max, var, scoring
    # -----------------------------------------------------------------------
    # Somme : somme des scores des 3 sources externes
    data['EXT_SOURCE_SUM'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                   'EXT_SOURCE_3']].sum(axis=1)
    # Moyenne : moyenne des scores des 3 sources externes
    data['EXT_SOURCE_MEAN'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                    'EXT_SOURCE_3']].mean(axis=1)
    # Multiplication : des scores des 3 sources externes
    data['EXT_SOURCE_MUL'] = data['EXT_SOURCE_1'] * \
        data['EXT_SOURCE_2'] * data['EXT_SOURCE_3']
    # Max : Max parmi les 3 scores des 3 sources externes
    data['EXT_SOURCE_MAX'] = [max(ele1, ele2, ele3) for ele1, ele2, ele3 in zip(
        data['EXT_SOURCE_1'], data['EXT_SOURCE_2'], data['EXT_SOURCE_3'])]
    # Min : Min parmi les 3 scores des 3 sources externes
    data['EXT_SOURCE_MIN'] = [min(ele1, ele2, ele3) for ele1, ele2, ele3 in zip(
        data['EXT_SOURCE_1'], data['EXT_SOURCE_2'], data['EXT_SOURCE_3'])]
    # Variance : variance des scores des 3 sources externes
    data['EXT_SOURCE_VAR'] = [np.var([ele1, ele2, ele3]) for ele1, ele2, ele3 in zip(
        data['EXT_SOURCE_1'], data['EXT_SOURCE_2'], data['EXT_SOURCE_3'])]
    # Scoring : scoring des scores des 3 sources externes, score 1 poids 2...
    data['WEIGHTED_EXT_SOURCE'] = data.EXT_SOURCE_1 * \
        2 + data.EXT_SOURCE_2 * 3 + data.EXT_SOURCE_3 * 4

    # -----------------------------------------------------------------------
    # Variables sur le bâtiment
    # -----------------------------------------------------------------------
    # Somme : Informations normalisées sur l'immeuble où vit le demandeur des moyennes
    # de la taille de l'appartement, de la surface commune, de la surface habitable,
    # de l'âge de l'immeuble, du nombre d'ascenseurs, du nombre d'entrées,
    # de l'état de l'immeuble et du nombre d'étages.
    data['APARTMENTS_SUM_AVG'] = data['APARTMENTS_AVG'] + data['BASEMENTAREA_AVG'] + data['YEARS_BEGINEXPLUATATION_AVG'] + data[
        'YEARS_BUILD_AVG'] + data['ELEVATORS_AVG'] + data['ENTRANCES_AVG'] + data[
        'FLOORSMAX_AVG'] + data['FLOORSMIN_AVG'] + data['LANDAREA_AVG'] + data[
        'LIVINGAREA_AVG'] + data['NONLIVINGAREA_AVG']
    # Somme : Informations normalisées sur l'immeuble où vit le demandeur des modes
    # de la taille de l'appartement, de la surface commune, de la surface habitable,
    # de l'âge de l'immeuble, du nombre d'ascenseurs, du nombre d'entrées,
    # de l'état de l'immeuble et du nombre d'étages.
    data['APARTMENTS_SUM_MODE'] = data['APARTMENTS_MODE'] + data['BASEMENTAREA_MODE'] + data['YEARS_BEGINEXPLUATATION_MODE'] + data[
        'YEARS_BUILD_MODE'] + data['ELEVATORS_MODE'] + data['ENTRANCES_MODE'] + data[
        'FLOORSMAX_MODE'] + data['FLOORSMIN_MODE'] + data['LANDAREA_MODE'] + data[
        'LIVINGAREA_MODE'] + data['NONLIVINGAREA_MODE'] + data['TOTALAREA_MODE']
    # Somme : Informations normalisées sur l'immeuble où vit le demandeur des médianes
    # de la taille de l'appartement, de la surface commune, de la surface habitable,
    # de l'âge de l'immeuble, du nombre d'ascenseurs, du nombre d'entrées,
    # de l'état de l'immeuble et du nombre d'étages.
    data['APARTMENTS_SUM_MEDI'] = data['APARTMENTS_MEDI'] + data['BASEMENTAREA_MEDI'] + data['YEARS_BEGINEXPLUATATION_MEDI'] + data[
        'YEARS_BUILD_MEDI'] + data['ELEVATORS_MEDI'] + data['ENTRANCES_MEDI'] + data[
        'FLOORSMAX_MEDI'] + data['FLOORSMIN_MEDI'] + data['LANDAREA_MEDI'] + \
        data['NONLIVINGAREA_MEDI']
    # Multiplication : somme des moyennes des infos sur le bâtiment * revenu
    # du demandeur
    data['INCOME_APARTMENT_AVG_MUL'] = data['APARTMENTS_SUM_AVG'] * \
        data['AMT_INCOME_TOTAL']
    # Multiplication : somme des modes des infos sur le bâtiment * revenu du
    # demandeur
    data['INCOME_APARTMENT_MODE_MUL'] = data['APARTMENTS_SUM_MODE'] * \
        data['AMT_INCOME_TOTAL']
    # Multiplication : somme des médianes des infos sur le bâtiment * revenu
    # du demandeur
    data['INCOME_APARTMENT_MEDI_MUL'] = data['APARTMENTS_SUM_MEDI'] * \
        data['AMT_INCOME_TOTAL']

    # -----------------------------------------------------------------------
    # Variables sur les défauts de paiements et les défauts observables
    # -----------------------------------------------------------------------
    # Somme : nombre d'observations de l'environnement social du demandeur
    #         avec des défauts observables de 30 DPD (jours de retard) +
    #        nombre d'observations de l'environnement social du demandeur
    #         avec des défauts observables de 60 DPD (jours de retard)
    data['OBS_30_60_SUM'] = data['OBS_30_CNT_SOCIAL_CIRCLE'] + \
        data['OBS_60_CNT_SOCIAL_CIRCLE']
    # Somme : nombre d'observations de l'environnement social du demandeur
    #         avec des défauts de paiement de 30 DPD (jours de retard) +
    #        nombre d'observations de l'environnement social du demandeur
    #         avec des défauts de paiement de 60 DPD (jours de retard)
    data['DEF_30_60_SUM'] = data['DEF_30_CNT_SOCIAL_CIRCLE'] + \
        data['DEF_60_CNT_SOCIAL_CIRCLE']
    # Multiplication : nombre d'observations de l'environnement social du demandeur
    #         avec des défauts observables de 30 DPD (jours de retard) *
    #        nombre d'observations de l'environnement social du demandeur
    #         avec des défauts observables de 60 DPD (jours de retard)
    data['OBS_DEF_30_MUL'] = data['OBS_30_CNT_SOCIAL_CIRCLE'] * \
        data['DEF_30_CNT_SOCIAL_CIRCLE']
    # Multiplication : nombre d'observations de l'environnement social du demandeur
    #         avec des défauts de paiement de 30 DPD (jours de retard) *
    #        nombre d'observations de l'environnement social du demandeur
    #         avec des défauts de paiement de 60 DPD (jours de retard)
    data['OBS_DEF_60_MUL'] = data['OBS_60_CNT_SOCIAL_CIRCLE'] * \
        data['DEF_60_CNT_SOCIAL_CIRCLE']
    # Somme : nombre d'observations de l'environnement social du demandeur
    #         avec des défauts de paiement ou des défauts observables avec 30
    #         DPD (jours de retard) et 60 DPD.
    data['SUM_OBS_DEF_ALL'] = data['OBS_30_CNT_SOCIAL_CIRCLE'] + data['DEF_30_CNT_SOCIAL_CIRCLE'] + \
        data['OBS_60_CNT_SOCIAL_CIRCLE'] + data['DEF_60_CNT_SOCIAL_CIRCLE']
    # Ratio : Montant du crédit du prêt /
    #         nombre d'observations de l'environnement social du demandeur
    #         avec des défauts observables de 30 DPD (jours de retard)
    data['OBS_30_CREDIT_RATIO'] = data['AMT_CREDIT'] / \
        (data['OBS_30_CNT_SOCIAL_CIRCLE'] + 0.00001)
    # Ratio : Montant du crédit du prêt /
    #         nombre d'observations de l'environnement social du demandeur
    #         avec des défauts observables de 60 DPD (jours de retard)
    data['OBS_60_CREDIT_RATIO'] = data['AMT_CREDIT'] / \
        (data['OBS_60_CNT_SOCIAL_CIRCLE'] + 0.00001)
    # Ratio : Montant du crédit du prêt /
    #         nombre d'observations de l'environnement social du demandeur
    #         avec des défauts de paiement de 30 DPD (jours de retard)
    data['DEF_30_CREDIT_RATIO'] = data['AMT_CREDIT'] / \
        (data['DEF_30_CNT_SOCIAL_CIRCLE'] + 0.00001)
    # Ratio : Montant du crédit du prêt /
    #         nombre d'observations de l'environnement social du demandeur
    #         avec des défauts de paiement de 60 DPD (jours de retard)
    data['DEF_60_CREDIT_RATIO'] = data['AMT_CREDIT'] / \
        (data['DEF_60_CNT_SOCIAL_CIRCLE'] + 0.00001)

    # -----------------------------------------------------------------------
    # Variables sur les indicateurs des documents fournis ou non
    # -----------------------------------------------------------------------
    # Toutes les variables DOCUMENT_
    cols_flag_doc = [flag for flag in data.columns if 'FLAG_DOC' in flag]
    # Somme : tous les indicateurs des documents fournis ou non
    data['FLAGS_DOCUMENTS_SUM'] = data[cols_flag_doc].sum(axis=1)
    # Moyenne : tous les indicateurs des documents fournis ou non
    data['FLAGS_DOCUMENTS_AVG'] = data[cols_flag_doc].mean(axis=1)
    # Variance : tous les indicateurs des documents fournis ou non
    data['FLAGS_DOCUMENTS_VAR'] = data[cols_flag_doc].var(axis=1)
    # Ecart-type : tous les indicateurs des documents fournis ou non
    data['FLAGS_DOCUMENTS_STD'] = data[cols_flag_doc].std(axis=1)

    # -----------------------------------------------------------------------
    # Variables sur le détail des modifications du demandeur : jour/heure...
    # -----------------------------------------------------------------------
    # Somme : nombre de jours avant la demande de changement de téléphone
    #         + nombre de jours avant la demande de changement enregistré sur la demande
    #         + nombre de jours avant la demande le client où il à
    #           changé la pièce d'identité avec laquelle il a demandé le prêt
    data['DAYS_DETAILS_CHANGE_SUM'] = data['DAYS_LAST_PHONE_CHANGE'] + \
        data['DAYS_REGISTRATION'] + data['DAYS_ID_PUBLISH']
    # Somme : nombre de demandes de renseignements sur le client adressées au Bureau de crédit
    # une heure + 1 jour + 1 mois + 3 mois + 1 an et 1 jour avant la demande
    data['AMT_ENQ_SUM'] = data['AMT_REQ_CREDIT_BUREAU_HOUR'] + data['AMT_REQ_CREDIT_BUREAU_DAY'] + data['AMT_REQ_CREDIT_BUREAU_WEEK'] + \
        data['AMT_REQ_CREDIT_BUREAU_MON'] + \
            data['AMT_REQ_CREDIT_BUREAU_QRT'] + \
                data['AMT_REQ_CREDIT_BUREAU_YEAR']
    # Ratio : somme du nombre de demandes de renseignements sur le client adressées au Bureau de crédit
    #         une heure + 1 jour + 1 mois + 3 mois + 1 an et 1 jour avant la demande \
    #         Montant du crédit du prêt
    data['ENQ_CREDIT_RATIO'] = data['AMT_ENQ_SUM'] / \
        (data['AMT_CREDIT'] + 0.00001)

    return data


# --------------------------------------------------------------------
# -- FEATURE ENGINEERING : super variable gagnant concours kaggle
# --------------------------------------------------------------------


def feature_engineering_neighbors_EXT_SOURCE(dataframe):
    '''
     - Imputation de la moyenne des 500 valeurs cibles des voisins les plus
       proches pour chaque application du train set ou test set.
     - Les voisins sont calculés en utilisant :
       - les variables très importantes :
       - EXT_SOURCE-1,
       - EXT_SOURCE_2
       - et EXT_SOURCE_3,
     - et CREDIT_ANNUITY_RATIO (ratio du Montant du crédit du prêt / Annuité de prêt).
     [Source](https://www.kaggle.com/c/home-credit-default-risk/discussion/64821)
     Inputs: dataframe pour lequel on veut ajouter la variable des 500 plus
             proches voisins.
     Returns:
         None
     '''

    knn = KNeighborsClassifier(500, n_jobs=-1)

    train_data_for_neighbors = dataframe[['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                          'EXT_SOURCE_3',
                                          'CREDIT_ANNUITY_RATIO']].fillna(0)

    # saving the training data for neighbors
    with open('../sauvegarde/pre-processing/TARGET_MEAN_500_Neighbors_training_data.pkl', 'wb') as f:
         pickle.dump(train_data_for_neighbors, f)
    train_target = dataframe.TARGET

    knn.fit(train_data_for_neighbors, train_target)
    # pickling the knn model
    with open('../sauvegarde/pre-processing/KNN_model_TARGET_500_neighbors.pkl', 'wb') as f:
         pickle.dump(knn, f)

    train_500_neighbors = knn.kneighbors(train_data_for_neighbors)[1]

    # adding the means of targets of 500 neighbors to new column
    dataframe['TARGET_NEIGHBORS_500_MEAN'] = [
        dataframe['TARGET'].iloc[ele].mean() for ele in train_500_neighbors]


# --------------------------------------------------------------------
# -- FEATURE ENGINEERING : super variable gagnant concours kaggle
# --------------------------------------------------------------------


def feature_engineering_neighbors_EXT_SOURCE_test(application_train, application_test):
    '''
     - Imputation de la moyenne des 500 valeurs cibles des voisins les plus
       proches pour chaque application du train set ou test set.
     - Les voisins sont calculés en utilisant :
       - les variables très importantes :
       - EXT_SOURCE-1,
       - EXT_SOURCE_2
       - et EXT_SOURCE_3,
     - et CREDIT_ANNUITY_RATIO (ratio du Montant du crédit du prêt / Annuité de prêt).
     [Source](https://www.kaggle.com/c/home-credit-default-risk/discussion/64821)
     Inputs: dataframe pour lequel on veut ajouter la variable des 500 plus
             proches voisins.
     Returns:
         None
     '''

    knn = KNeighborsClassifier(500, n_jobs=-1)

    train_data_for_neighbors = application_train[[
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'CREDIT_ANNUITY_RATIO'
    ]].fillna(0)

    train_target = application_train.TARGET
    test_data_for_neighbors = application_test[[
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'CREDIT_ANNUITY_RATIO'
    ]].fillna(0)

    knn.fit(train_data_for_neighbors, train_target)

    test_500_neighbors = knn.kneighbors(test_data_for_neighbors)[1]

    application_test['TARGET_NEIGHBORS_500_MEAN'] = [
        application_train['TARGET'].iloc[ele].mean()
        for ele in test_500_neighbors
    ]
    

# --------------------------------------------------------------------
# -- AGGREGATION DES VARIABLES STATISTIQUES des VAR QUANTITATIVES
# --------------------------------------------------------------------

def agg_var_num(dataframe, group_var, dict_agg, prefix):
    """
    Aggregates the numeric values in a dataframe.
    This can be used to create features for each instance of the grouping variable.
    Parameters
    --------
        dataframe (dataframe): the dataframe to calculate the statistics on
        group_var (string): the variable by which to group df
        df_name (string): the variable used to rename the columns
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            some statistics (mean, min, max, sum ...) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    # Remove id variables other than grouping variable
    for col in dataframe:
        if col != group_var and 'SK_ID' in col:
            dataframe = dataframe.drop(columns=col)

    group_ids = dataframe[group_var]
    numeric_df = dataframe.select_dtypes('number')
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(dict_agg)

    # Ajout suffix mean, sum...
    agg.columns = ['_'.join(tup).strip().upper()
                   for tup in agg.columns.values]

    # Ajout du prefix bureau_balance pour avoir une idée du fichier
    agg.columns = [prefix + '_' + col
                   if col != group_var else col
                   for col in agg.columns]

    agg.reset_index(inplace=True)

    return agg


# --------------------------------------------------------------------
# -- AGGREGATION DES VARIABLES STATISTIQUES des VAR QUALITATIVES
# --------------------------------------------------------------------

def agg_var_cat(dataframe, group_var, prefix):
    '''
        Aggregates the categorical features in a child dataframe
        for each observation of the parent variable.
        
        Parameters
        --------
        - dataframe        : pandas dataframe
                    The dataframe to calculate the value counts for.
            
        - parent_var : string
                    The variable by which to group and aggregate 
                    the dataframe. For each unique value of this variable, 
                    the final dataframe will have one row
            
        - prefix    : string
                    Variable added to the front of column names 
                    to keep track of columns

        Return
        --------
        categorical : pandas dataframe
                    A dataframe with aggregated statistics for each observation 
                    of the parent_var
                    The columns are also renamed and columns with duplicate values 
                    are removed.
    '''
    
    # Select the categorical columns
    categorical = pd.get_dummies(dataframe.select_dtypes('object'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = dataframe[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'count', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['sum', 'count', 'mean']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (prefix, var, stat))
    
    categorical.columns = column_names
    
    # Remove duplicate columns by values
    # _, idx = np.unique(categorical, axis = 1, return_index = True)
    # categorical = categorical.iloc[:, idx]
    
    return categorical

# --------------------------------------------------------------------
# -- AGGREGATION DES VARIABLES STATISTIQUES des VAR QUANTITATIVES
# -- PAR MOYENNE PAR SK_ID_CURR de prêts
# --------------------------------------------------------------------

def agg_moy_par_pret(dataframe, group_var, prefix):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    --------
        dataframe (dataframe): 
            the dataframe to calculate the statistics on
        group_var (string): 
            the variable by which to group df
        prefix (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    # Remove id variables other than grouping variable
    for col in dataframe:
        if col != group_var and 'SK_ID' in col:
            dataframe = dataframe.drop(columns = col)
            
    group_ids = dataframe[group_var]
    numeric_df = dataframe.select_dtypes('number')
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['mean']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (prefix, var, stat))

    agg.columns = columns
    
    return agg

# --------------------------------------------------------------------
# -- GESTION DES VARIABLES FORTEMENT COLINEAIRES
# --------------------------------------------------------------------

def suppr_var_colineaire(dataframe, seuil=0.8):
    '''
    Récupération de la liste des variables fortement corrélées supérieur
    au seuil transmis.
    Parameters
    ----------
    dataframe : dataframe à analyser, obligatoire.
    seuil : le seuil de colinéarité entre les variables (0.8 par défaut).
    Returns
    -------
    cols_corr_a_supp : liste des variables à supprimer.
    '''
    
    # Matrice de corrélation avec valeur absolue pour ne pas avoir à gérer
    # les corrélations positives et négatives séparément
    corr = dataframe.corr().abs()
    # On ne conserve que la partie supérieur à la diagonale pour n'avoir
    # qu'une seule fois les corrélations prisent en compte (symétrie axiale)
    corr_triangle = corr.where(np.triu(np.ones(corr.shape), k=1)
                               .astype(np.bool))
    
    # Variables avec un coef de Pearson > 0.8?
    cols_corr_a_supp = [var for var in corr_triangle.columns
                        if any(corr_triangle[var] > seuil)]
    print(f'{len(cols_corr_a_supp)} variables fortement corrélées à supprimer :\n')
    for var in cols_corr_a_supp:
        print(var)
        
    return cols_corr_a_supp

# --------------------------------------------------------------------
# -- FEATURES SELECTION AVEC BORUTA ET MODELE RANDOMFOREST
# --------------------------------------------------------------------

def features_selection_boruta(dataframe, titre):
    '''
    
    Parameters
    ----------
    dataframe : dataframe dont on veut extraire les features importances
                avec boruta, obligatoire.
    Returns
    -------
    df_fs_boruta : liste des variables avec haute importance selon boruta.

    '''
    # Sauvegarde des étiquettes
    dataframe_labels = dataframe['TARGET']
    
    # Suppression des identifiants (variable non utile pour les variables
    # pertinentes)
    dataframe = dataframe.drop(columns=['SK_ID_CURR'])
    dataframe = dataframe.drop(columns=['TARGET'])
    print(f'train_fs_boruta : {dataframe.shape}')
    
    # Initialisation des variables
    X = dataframe.values
    y = dataframe_labels.values.ravel()
    
    rf = RandomForestClassifier(n_jobs=-1,
                            class_weight='balanced',
                            max_depth=5)
    
    # Initialisation de Boruta
    boruta_feature_selector = BorutaPy(rf,
                                       n_estimators='auto',
                                       verbose=2,
                                       random_state=21,
                                       max_iter=50,
                                       perc=90)
    # Entraînement
    boruta_feature_selector.fit(X, y)
    
    # On applique le modèle sur le dataset
    X_filtered = boruta_feature_selector.transform(X)
    print(f'X_transform : {X_filtered.shape}')
    
    # Liste des variables confirmées avec une haute importance
    fs_boruta = list()
    features = [f for f in dataframe.columns]
    indexes = np.where(boruta_feature_selector.support_ == True)
    for x in np.nditer(indexes):
        fs_boruta.append(features[x])
    print(f'fs_boruta : {fs_boruta}') 
    
    # Dataframe de features importance avec boruta
    df_fs_boruta = pd.DataFrame(fs_boruta)
    
    # Sauvegarde des features importances avec boruta
    fic_sav_fs_boruta = \
        '../sauvegarde/features-selection/' + titre + '.pickle'
    with open(fic_sav_fs_boruta, 'wb') as f:
        pickle.dump(df_fs_boruta, f, pickle.HIGHEST_PROTOCOL)
    
    return df_fs_boruta


# --------------------------------------------------------------------
# -- FEATURES SELECTION AVEC BORUTA ET MODELE LIGHTGBM
# --------------------------------------------------------------------

def features_selection_boruta_lgbm(dataframe, titre):
    '''
    
    Parameters
    ----------
    dataframe : dataframe dont on veut extraire les features importances
                avec boruta, obligatoire.
    Returns
    -------
    df_fs_boruta : liste des variables avec haute importance selon boruta.

    '''
    # Sauvegarde des étiquettes
    dataframe_labels = dataframe['TARGET']
    
    # Suppression des identifiants (variable non utile pour les variables
    # pertinentes)
    dataframe = dataframe.drop(columns=['SK_ID_CURR'])
    dataframe = dataframe.drop(columns=['TARGET'])
    print(f'train_fs_boruta : {dataframe.shape}')
    
    # Initialisation des variables
    X = dataframe.values
    y = dataframe_labels.values.ravel()
    
    # Create the model with several hyperparameters
    lgbm = lgb.LGBMClassifier(objective='binary',
                              boosting_type='goss',
                              n_estimators=10000,
                              class_weight='balanced',
                              num_boost_round=100)
    
    # Initialisation de Boruta
    boruta_feature_selector = BorutaPy(lgbm,
                                       n_estimators='auto',
                                       verbose=2,
                                       random_state=21,
                                       max_iter=50,
                                       perc=90)
    # Entraînement
    boruta_feature_selector.fit(X, y)
    
    # On applique le modèle sur le dataset
    X_filtered = boruta_feature_selector.transform(X)
    print(f'X_transform : {X_filtered.shape}')
    
    # Liste des variables confirmées avec une haute importance
    fs_boruta = list()
    features = [f for f in dataframe.columns]
    indexes = np.where(boruta_feature_selector.support_ == True)
    for x in np.nditer(indexes):
        fs_boruta.append(features[x])
    print(f'fs_boruta : {fs_boruta}') 
    
    # Dataframe de features importance avec boruta
    df_fs_boruta = pd.DataFrame(fs_boruta)
    
    # Sauvegarde des features importances avec boruta
    fic_sav_fs_boruta = \
        '../sauvegarde/features-selection/' + titre + '.pickle'
    with open(fic_sav_fs_boruta, 'wb') as f:
        pickle.dump(df_fs_boruta, f, pickle.HIGHEST_PROTOCOL)
    
    return df_fs_boruta


class BorutaPyForLGB(BorutaPy):
    def __init__(self, estimator, n_estimators=1000, perc=100, alpha=0.05,
                 two_step=True, max_iter=100, random_state=None, verbose=0):
        super().__init__(estimator, n_estimators, perc, alpha,
                         two_step, max_iter, random_state, verbose)
        self._is_lightgbm = 'lightgbm' in str(type(self.estimator))
        
    def _fit(self, X, y):
        # check input params
        self._check_params(X, y)

        if not isinstance(X, np.ndarray):
            X = self._validate_pandas_input(X) 
        if not isinstance(y, np.ndarray):
            y = self._validate_pandas_input(y)

        self.random_state = check_random_state(self.random_state)
        # setup variables for Boruta
        n_sample, n_feat = X.shape
        _iter = 1
        # holds the decision about each feature:
        # 0  - default state = tentative in original code
        # 1  - accepted in original code
        # -1 - rejected in original code
        dec_reg = np.zeros(n_feat, dtype=np.int)
        # counts how many times a given feature was more important than
        # the best of the shadow features
        hit_reg = np.zeros(n_feat, dtype=np.int)
        # these record the history of the iterations
        imp_history = np.zeros(n_feat, dtype=np.float)
        sha_max_history = []

        # set n_estimators
        if self.n_estimators != 'auto':
            self.estimator.set_params(n_estimators=self.n_estimators)

        # main feature selection loop
        while np.any(dec_reg == 0) and _iter < self.max_iter:
            # find optimal number of trees and depth
            if self.n_estimators == 'auto':
                # number of features that aren't rejected
                not_rejected = np.where(dec_reg >= 0)[0].shape[0]
                n_tree = self._get_tree_num(not_rejected)
                self.estimator.set_params(n_estimators=n_tree)

            # make sure we start with a new tree in each iteration
            if self._is_lightgbm:
                self.estimator.set_params(random_state=self.random_state.randint(0, 10000))
            else:
                self.estimator.set_params(random_state=self.random_state)

            # add shadow attributes, shuffle them and train estimator, get imps
            cur_imp = self._add_shadows_get_imps(X, y, dec_reg)

            # get the threshold of shadow importances we will use for rejection
            imp_sha_max = np.percentile(cur_imp[1], self.perc)

            # record importance history
            sha_max_history.append(imp_sha_max)
            imp_history = np.vstack((imp_history, cur_imp[0]))

            # register which feature is more imp than the max of shadows
            hit_reg = self._assign_hits(hit_reg, cur_imp, imp_sha_max)

            # based on hit_reg we check if a feature is doing better than
            # expected by chance
            dec_reg = self._do_tests(dec_reg, hit_reg, _iter)

            # print out confirmed features
            if self.verbose > 0 and _iter < self.max_iter:
                self._print_results(dec_reg, _iter, 0)
            if _iter < self.max_iter:
                _iter += 1

        # we automatically apply R package's rough fix for tentative ones
        confirmed = np.where(dec_reg == 1)[0]
        tentative = np.where(dec_reg == 0)[0]
        # ignore the first row of zeros
        tentative_median = np.median(imp_history[1:, tentative], axis=0)
        # which tentative to keep
        tentative_confirmed = np.where(tentative_median
                                       > np.median(sha_max_history))[0]
        tentative = tentative[tentative_confirmed]

        # basic result variables
        self.n_features_ = confirmed.shape[0]
        self.support_ = np.zeros(n_feat, dtype=np.bool)
        self.support_[confirmed] = 1
        self.support_weak_ = np.zeros(n_feat, dtype=np.bool)
        self.support_weak_[tentative] = 1

        # ranking, confirmed variables are rank 1
        self.ranking_ = np.ones(n_feat, dtype=np.int)
        # tentative variables are rank 2
        self.ranking_[tentative] = 2
        # selected = confirmed and tentative
        selected = np.hstack((confirmed, tentative))
        # all rejected features are sorted by importance history
        not_selected = np.setdiff1d(np.arange(n_feat), selected)
        # large importance values should rank higher = lower ranks -> *(-1)
        imp_history_rejected = imp_history[1:, not_selected] * -1

        # update rank for not_selected features
        if not_selected.shape[0] > 0:
                # calculate ranks in each iteration, then median of ranks across feats
                iter_ranks = self._nanrankdata(imp_history_rejected, axis=1)
                rank_medians = np.nanmedian(iter_ranks, axis=0)
                ranks = self._nanrankdata(rank_medians, axis=0)

                # set smallest rank to 3 if there are tentative feats
                if tentative.shape[0] > 0:
                    ranks = ranks - np.min(ranks) + 3
                else:
                    # and 2 otherwise
                    ranks = ranks - np.min(ranks) + 2
                self.ranking_[not_selected] = ranks
        else:
            # all are selected, thus we set feature supports to True
            self.support_ = np.ones(n_feat, dtype=np.bool)

        self.importance_history_ = imp_history

        # notify user
        if self.verbose > 0:
            self._print_results(dec_reg, _iter, 1)
        return self
    

def tracer_features_importance(dataframe, df_features_importance, jeu, methode):
    """
    Affiche l'étape puis nombre de lignes et de variables pour le dataframe transmis
    Parameters
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                df_features_importance : dataframe de suivi des dimensions,
                                         obligatoire
                jeu : jeu de données train_set, train set avec imputation 1...
                methode : titre du modèle de feature sélection
    @param OUT : dataframe de suivi des dimensions
    """
    # Nombre de variables retenues lors de la feature selection
    n_features = dataframe.shape[0]
    print(f'{jeu} - {methode} : {n_features} variables importantes conservées')

    df_features_importance = \
        df_features_importance.append({'Jeu_données': jeu,
                                       'Méthode': methode,
                                       'Nb_var_importante': n_features},
                                       ignore_index=True)

    # Suivi dimensions
    return df_features_importance