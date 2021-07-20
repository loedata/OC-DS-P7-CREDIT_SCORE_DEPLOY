""" Librairie personnelle pour exécuter des tests de normalité,
    homostéradiscité, ANOVA, Kruskall-Wallis
"""

#! /usr/bin/env python3
# coding: utf-8

# ====================================================================
# Outil visualisation -  projet 3 Openclassrooms
# Version : 0.0.0 - CRE LR 13/03/2021
# ====================================================================

from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats import shapiro, normaltest, anderson
import pandas as pd
from IPython.display import display

# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.0'

# --------------------------------------------------------------------
# -- TESTS DE NORMALITE
# --------------------------------------------------------------------


def test_normalite(data):
    """
    Test de la normalité d'une distribution.
    Parameters
    ----------
    data : dataframe ou dataframe restreint (une seule variable) obligatoire
    Returns
    -------
    None.
    """
    #  H0 : la distribution des données est normale (P>0,05)
    #  H1 : la distribution des données n'est pas normale (P<0,05)

    df_resultat = pd.DataFrame([])
    # Shapiro-Wilk - D'Agostino's K^2
    for f_name, func in zip(
            ['Shapiro-Wilks', "D'Agostino K^2"], [shapiro, normaltest]):
        stat, p_val = func(data)
        df_resultat.loc[f_name, 'stat'] = stat
        df_resultat.loc[f_name, 'p_value'] = p_val
        df_resultat.loc[f_name, 'res'] = [p_val > 0.05]
        bool = df_resultat.loc[f_name, 'res']
        print
        if bool:
            df_resultat.loc[f_name,
                            'bilan'] = 'H0 aceptée - distribution normale'
        else:
            df_resultat.loc[f_name,
                            'bilan'] = 'H0 rejetée - distribution non normale'

    # Anderson-Darling
    result = anderson(data, dist='norm')
    df_resultat.loc['Anderson-Darling', 'stat'] = result.statistic
    res_and = [(int(result.significance_level[i]), result.statistic < res)
               for i, res in enumerate(result.critical_values)]
    df_resultat.loc['Anderson-Darling', 'res'] = str(res_and)
    display(df_resultat)


# --------------------------------------------------------------------
# -- TESTS D'INDEPENDANCE DE 2 VARIABLES QUALITATIVES
# --------------------------------------------------------------------


def test_chi2(serie1, serie2):
    """
        Test de dépendances de 2 variables qualitatives
        Parameters
        ----------
        serie1 : variable qualitative 1 obligatoire
        serie2 : variable qualitative 2 obligatoire
        Returns
        -------
        None.
    """
    alpha = 0.03

    # H0 : les variables sont indépendantes

    #print('tableau de contingence :\n', pd.crosstab(serie1.array, serie2.array))
    tab_contingence = pd.crosstab(serie1.array, serie2.array)
    stat_chi2, p_value, dof, expected_table = chi2_contingency(
        tab_contingence.values)
    print('chi2 : {0:.5f}'.format(stat_chi2))
    print('\np_value : {0:.5f}'.format(p_value))
    print('\ndof : {0:.5f}\n'.format(dof))
    critical = chi2.ppf(1 - alpha, dof)
    print('critical : ', critical)

    if p_value <= alpha:
        print(
            '\nVariables non indépendantes (H0 rejetée) car p_value = {} <= alpha = {}'.format(
                p_value,
                alpha))
    else:
        print('\nH0 non rejetée car p = {} >= alpha = {}'.format(p_value, alpha))


# -----------------------------------------------------------------------
# -- TESTS D'INDEPENDANCE ENTRE 1 VARIABLE QUANTITATIVE ET 1 QUALITATIVE
# -----------------------------------------------------------------------


def test_eta_squared(serie_qualitative, serie_quantitative):
    """
        Test de dépendances de 1 variable qualitative et 1 quantitative
        Parameters
        ----------
        serie_qualitative : variable qualitative, obligatoire
        serie_quantitative : variable quantitative, obligatoire
        Returns
        -------
        eta_squared.
    """
    moyenne_y = serie_quantitative.mean()
    classes = []
    for classe in serie_qualitative.unique():
        yi_classe = serie_quantitative[serie_qualitative == classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj - moyenne_y)**2 for yj in serie_quantitative])
    SCE = sum([c['ni'] * (c['moyenne_classe'] - moyenne_y)**2 for c in classes])

    return SCE / SCT

# X = "categ" # qualitative
# Y = "montant" # quantitative

# def eta_squared(x,y):

# eta_squared(sous_echantillon[X],sous_echantillon[Y])
