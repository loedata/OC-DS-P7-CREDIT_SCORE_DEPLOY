""" Librairie contenant les fonctions utilisees lors de l'analyse exploratoire
    provenant du github de Rishabh Rao :
     https://github.com/rishabhrao1997/Home-Credit-Default-Risk
    Pour les explications :
     https://medium.com/thecyphy/home-credit-default-risk-part-1-3bfe3c7ddd7a
"""

#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# ====================================================================
# Outils Fonctions EDA Kernel Kaggle -  projet 7 Openclassrooms
# Version : 0.0.0 - CRE LR 07/07/2021
# ====================================================================
from IPython.core.display import display
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import phik

# Plotly
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plotly.offline.init_notebook_mode(connected=True)

# for 100% jupyter notebook cell width
# display(HTML("<style>.container { width:100% !important; }</style>"))

# pandas DataFrame column and row display limits
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)


# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.0'


# --------------------------------------------------------------------
# -- CHARGEMENT DES JEUX DE DONNÉES
# --------------------------------------------------------------------


def load_all_tables(directory_path='', verbose=True):
    '''
    Function to load all the tables required
    Input:
        directory_path: str, default = ''
            Path of directory in which tables are stored in
        verbose: bool, default = True
            Whether to keep verbosity or not

    '''
    if verbose:
        print("Chargement des jeux de donnees...")
        print("--------------------------------------------------------------------")
        start = datetime.now()

    application_train = pd.read_csv(directory_path + 'application_train.csv')
    if verbose:
        print("Fichier application_train.csv chargé -> dataframe : application_train")

    application_test = pd.read_csv(directory_path + 'application_test.csv')
    if verbose:
        print("Fichier application_test.csv chargé -> dataframe : application_test")

    bureau = pd.read_csv(directory_path + 'bureau.csv')
    if verbose:
        print("Fichier bureau.csv chargé -> dataframe : bureau")

    bureau_balance = pd.read_csv(directory_path + 'bureau_balance.csv')
    if verbose:
        print("Fichier bureau_balance.csv chargé -> dataframe : bureau_balance")

    cc_balance = pd.read_csv(directory_path + 'credit_card_balance.csv')
    if verbose:
        print("Fichier credit_card_balance.csv chargé -> dataframe : cc_balance")

    installments_payments = pd.read_csv(
        directory_path + 'installments_payments.csv')
    if verbose:
        print(
            "Fichier installments_payments.csv chargé -> dataframe : installments_payments")

    POS_CASH_balance = pd.read_csv(directory_path + 'POS_CASH_balance.csv')
    if verbose:
        print("Fichier POS_CASH_balance.csv chargé -> dataframe : POS_CASH_balance")

    HomeCredit_columns_description = pd.read_csv(
        directory_path +
        'HomeCredit_columns_description.csv',
        encoding='cp1252')
    del HomeCredit_columns_description['Unnamed: 0']
    if verbose:
        print("Fichier HomeCredit_columns_description.csv chargé -> dataframe : HomeCredit_columns_description")

    previous_application = pd.read_csv(
        directory_path + 'previous_application.csv')
    if verbose:
        print("Fichier previous_application.csv chargé -> dataframe : previous_application")

    if verbose:
        print("--------------------------------------------------------------------")
        print(
            f'Chargement des 9 jeux de donnees terminée en {datetime.now() - start} secondes')

    return application_train, application_test, bureau, bureau_balance, \
        cc_balance, installments_payments, POS_CASH_balance, previous_application, \
        HomeCredit_columns_description


# --------------------------------------------------------------------
# -- AFFICHER LES STATISTIQUES SIMPLES ligne, varaible,duplicate, head
# --------------------------------------------------------------------


def afficher_stats_basic(dataframe, titre='application_train'):
    print('-' * 79)
    # Taille : nombre de lignes/colonnes
    nRow, nVar = dataframe.shape
    print(f'Le jeu de données contient {nRow} lignes et {nVar} variables.')
    print('-' * 79)
    duplicate = \
        dataframe.shape[0] - dataframe.duplicated().shape[0]
    print(f'Numbre de valeurs dupliquées dans {titre} : {duplicate}')
    print('-' * 79)
    print('Contenu des 3 premières lignes :')
    display(dataframe.head(3))


def afficher_stats_basic_bureau(dataframe, dataframe2, dataframe3):
    print('-' * 79)
    # Taille : nombre de lignes/colonnes
    nRow, nVar = dataframe.shape
    print(f'Le jeu de données contient {nRow} lignes et {nVar} variables.')
    print('-' * 79)
    print(f'Nombre de valeurs uniques SK_ID_BUREAU  : {len(dataframe.SK_ID_BUREAU.unique())}')
    print(f'Nombre de valeurs uniques SK_ID_CURR dans bureau.csv : {len(dataframe.SK_ID_CURR.unique())}')
    print(f'Nombre de valeurs communes SK_ID_CURR dans application_train.csv et dataframe.csv : {len(set(dataframe2.SK_ID_CURR.unique()).intersection(set(dataframe.SK_ID_CURR.unique())))}')
    print(f'Nombre de valeurs communes SK_ID_CURR dans application_test.csv et dataframe.csv : {len(set(dataframe3.SK_ID_CURR.unique()).intersection(set(dataframe.SK_ID_CURR.unique())))}')
    print('-'*79)
    duplicate = \
        dataframe.shape[0] - dataframe.duplicated().shape[0]
    print(f'Numbre de valeurs dupliquées dans bureau : {duplicate}')
    print('-' * 79)
    print('Contenu des 3 premières lignes :')
    display(dataframe.head(5))    


def afficher_stats_basic_all(dataframe, dataframe2, dataframe3, id_cle):
    print('-' * 79)
    # Taille : nombre de lignes/colonnes
    nRow, nVar = dataframe.shape
    print(f'Le jeu de données contient {nRow} lignes et {nVar} variables.')
    print('-' * 79)
    print(f'Nombre de valeurs uniques {id_cle} dans {dataframe.name}.csv : {len(dataframe[id_cle].unique())}')
    print(f'Nombre de valeurs uniques SK_ID_CURR dans {dataframe.name}.csv : {len(dataframe.SK_ID_CURR.unique())}')
    print(f'Nombre de valeurs communes SK_ID_CURR dans {dataframe2.name}.csv et {dataframe.name}.csv : {len(set(dataframe2.SK_ID_CURR.unique()).intersection(set(dataframe.SK_ID_CURR.unique())))}')
    print(f'Nombre de valeurs communes SK_ID_CURR dans {dataframe2.name}.csv et {dataframe.name}.csv : {len(set(dataframe3.SK_ID_CURR.unique()).intersection(set(dataframe.SK_ID_CURR.unique())))}')
    print('-'*79)
    duplicate = \
        dataframe.shape[0] - dataframe.duplicated().shape[0]
    print(f'Numbre de valeurs dupliquées dans {dataframe.name}.csv : {duplicate}')
    print('-' * 79)
    print('Contenu des 3 premières lignes :')
    display(dataframe.head(5))    
    
    
# --------------------------------------------------------------------
# -- CREATION DATAFRAME DES VALEURS MANQUANTES
# --------------------------------------------------------------------


def nan_df_create(data):
    '''
    Function to create a dataframe of percentage of NaN values for each column of the dataframe
    Inputs:
        data:
            DataFrame

    Returns:
        DataFrame of NaN percentages
    '''
    nan_percentages = data.isna().sum() * 100 / len(data)
    df = pd.DataFrame({'column': nan_percentages.index,
                       'percent': nan_percentages.values})

    # sorting the dataframe by decreasing order of percentage of NaN values
    df.sort_values(by='percent', ascending=False, inplace=True)

    return df


# --------------------------------------------------------------------
# -- REPRESENTATION BARPLOT DES VALEURS MANQUANTES PAR VARIABLE
# --------------------------------------------------------------------


def plot_nan_percent(df_nan, title_name, tight_layout=True, figsize=(20, 8),
                     grid=False, rotation=90, fontsize=16):
    '''
    Function to plot Bar Plots of NaN percentages for each Column with missing values

    Inputs:
        df_nan:
            DataFrame of NaN percentages
        title_name:
            Name of table to be displayed in title of plot
        tight_layout: bool, default = True
            Whether to keep tight layout or not
        figsize: tuple, default = (20,8)
            Figure size of plot
        grid: bool, default = False
            Whether to draw gridlines to plot or not
        rotation: int, default = 0
            Degree of rotation for x-tick labels
    '''
    sns.set(style='white', font_scale=1.2)
    # checking if there is any column with NaNs or not.
    if df_nan.percent.sum() != 0:
        print(
            f"Nombre de variables avec valeurs manquantes : {df_nan[df_nan['percent'] != 0].shape[0]}")

        # plotting the Bar-Plot for NaN percentages (only for columns with
        # Non-Zero percentage of NaN values)
        plt.figure(figsize=figsize, tight_layout=tight_layout)
        sns.barplot(x='column', y='percent',
                    data=df_nan[df_nan['percent'] > 0])
        plt.xticks(rotation=rotation)
        plt.xlabel('Nom de variable', fontsize=fontsize)
        plt.ylabel('% de valeurs NaN', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(f'Pourcentage de valeurs NaN dans {title_name}', fontsize=fontsize+4)
        if grid:
            plt.grid()
        plt.show()
    else:
        print(
            f"Le dataframe {title_name} ne contient pas de valeurs manquantes.")


# --------------------------------------------------------------------
# -- CARTE THERMIQUE DES VALEURS DU COEFFICIENT DE CORRELATION Phi-K
# --------------------------------------------------------------------


def plot_phik_matrix(data, categorical_columns, figsize=(20, 20),
                     mask_upper=True, tight_layout=True, linewidth=0.1,
                     fontsize=10, cmap='Blues', show_target_top_corr=True,
                     target_top_columns=10):
    '''
    Function to Phi_k matrix for categorical features
    Nous allons tracer une carte thermique des valeurs du coefficient de
    corrélation Phi-K entre les 2 variables.
    Le coefficient Phi-K est similaire au coefficient de corrélation sauf
    qu'il peut être utilisé avec une paire de caractéristiques catégorielles
    pour vérifier si une varaible montre une sorte d'association avec l'autre
    variable catégorielle. Sa valeur maximale peut être de 1, ce qui indique
    une association maximale entre deux variables catégorielles.
    Inputs:
        data: DataFrame
            The DataFrame from which to build correlation matrix
        categorical_columns: list
            List of categorical columns whose PhiK values are to be plotted
        figsize: tuple, default = (25,23)
            Size of the figure to be plotted
        mask_upper: bool, default = True
            Whether to plot only the lower triangle of heatmap or plot full.
        tight_layout: bool, default = True
            Whether to keep tight layout or not
        linewidth: float/int, default = 0.1
            The linewidth to use for heatmap
        fontsize: int, default = 10
            The font size for the X and Y tick labels
        cmap: str, default = 'Blues'
            The colormap to be used for heatmap
        show_target_top_corr: bool, default = True
            Whether to show top/highly correlated features with Target.
        target_top_columns: int, default = 10
            The number of top correlated features with target to display
    '''
    # first fetching only the categorical features
    data_for_phik = data[categorical_columns].astype('object')
    phik_matrix = data_for_phik.phik_matrix()

    print('-' * 79)

    if mask_upper:
        mask_array = np.ones(phik_matrix.shape)
        mask_array = np.triu(mask_array)
    else:
        mask_array = np.zeros(phik_matrix.shape)

    plt.figure(figsize=figsize, tight_layout=tight_layout)
    sns.heatmap(
        phik_matrix,
        annot=False,
        mask=mask_array,
        linewidth=linewidth,
        cmap=cmap)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.title("Phi-K Correlation Heatmap des variables catégorielles",
              fontsize=fontsize+4)
    plt.show()
    
    print("-" * 79)

    if show_target_top_corr:
        # Seeing the top columns with highest correlation with the target
        # variable in application_train
        print("Les catégories présentant les valeurs les plus élevées de la corrélation Phi-K avec la variable cible sont les suivantes :")
        phik_df = pd.DataFrame(
            {'Variable': phik_matrix.TARGET.index[1:], 'Phik-Correlation': phik_matrix.TARGET.values[1:]})
        phik_df = phik_df.sort_values(by='Phik-Correlation', ascending=False)
        display(phik_df.head(target_top_columns).style.hide_index())
        print("-" * 79)


# --------------------------------------------------------------------
# -- MATRICE DE CORRELATION POUR LES VARIABLES NUMERIQUES
# --------------------------------------------------------------------


class correlation_matrix:
    '''
    Class to plot heatmap of Correlation Matrix and print Top Correlated Features with Target.
    Contains three methods:
        1. init method
        2. plot_correlation_matrix method
        3. target_top_corr method
    '''

    def __init__(
            self,
            data,
            columns_to_drop,
            figsize=(
                25,
                23),
            mask_upper=True,
            tight_layout=True,
            linewidth=0.1,
            fontsize=10,
            cmap='Blues'):
        '''
        Function to initialize the class members.

        Inputs:
            data: DataFrame
                The DataFrame from which to build correlation matrix
            columns_to_drop: list
                Columns which have to be dropped while building the correlation matrix (for example the Loan ID)
            figsize: tuple, default = (25,23)
                Size of the figure to be plotted
            mask_upper: bool, default = True
                Whether to plot only the lower triangle of heatmap or plot full.
            tight_layout: bool, default = True
                Whether to keep tight layout or not
            linewidth: float/int, default = 0.1
                The linewidth to use for heatmap
            fontsize: int, default = 10
                The font size for the X and Y tick labels
            cmap: str, default = 'Blues'
                The colormap to be used for heatmap

        Returns:
            None
        '''

        self.data = data
        self.columns_to_drop = columns_to_drop
        self.figsize = figsize
        self.mask_upper = mask_upper
        self.tight_layout = tight_layout
        self.linewidth = linewidth
        self.fontsize = fontsize
        self.cmap = cmap

    def plot_correlation_matrix(self):
        '''
        Function to plot the Correlation Matrix Heatmap

        Inputs:
            self

        Returns:
            None
        '''

        # print('-' * 79)
        # building the correlation dataframe
        self.corr_data = self.data.drop(
            self.columns_to_drop + ['TARGET'], axis=1).corr()

        if self.mask_upper:
            # masking the heatmap to show only lower triangle. This is to save
            # the RAM.
            mask_array = np.ones(self.corr_data.shape)
            mask_array = np.triu(mask_array)
        else:
            mask_array = np.zeros(self.corr_data.shape)

        plt.figure(figsize=self.figsize, tight_layout=self.tight_layout)
        sns.heatmap(
            self.corr_data,
            annot=False,
            mask=mask_array,
            linewidth=self.linewidth,
            cmap=self.cmap)
        plt.xticks(rotation=90, fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        plt.title("Heatmap de corrélation des variables numériques", fontsize=20)
        plt.show()
        # print("-" * 100)

    def target_top_corr(self, target_top_columns=10):
        '''
        Function to return the Top Correlated features with the Target

        Inputs:
            self
            target_top_columns: int, default = 10
                The number of top correlated features with target to display

        Returns:
            Top correlated features DataFrame.
        '''

        phik_target_arr = np.zeros(self.corr_data.shape[1])
        # calculating the Phik-Correlation with Target
        for index, column in enumerate(self.corr_data.columns):
            phik_target_arr[index] = self.data[[
                'TARGET', column]].phik_matrix().iloc[0, 1]
        # getting the top correlated columns and their values
        top_corr_target_df = pd.DataFrame(
            {'Column Name': self.corr_data.columns, 'Phik-Correlation': phik_target_arr})
        top_corr_target_df = top_corr_target_df.sort_values(
            by='Phik-Correlation', ascending=False)

        return top_corr_target_df.iloc[:target_top_columns]


# --------------------------------------------------------------------
# -- AFFICHE LA LISTE DES IDENTIFIANTS UNIQUES
# --------------------------------------------------------------------


def print_unique_categories(data, column_name, show_counts=False):
    '''
    Function to print the basic stats such as unique categories and their counts for categorical variables

        Inputs:
        data: DataFrame
            The DataFrame from which to print statistics
        column_name: str
            Column's name whose stats are to be printed
        show_counts: bool, default = False
            Whether to show counts of each category or not

    '''

    print('-' * 79)
    print(
        f"Les catégories uniques de la variable '{column_name}' sont :\n{data[column_name].unique()}")
    print('-' * 79)

    if show_counts:
        print(
            f"Répartition dans chaque catégorie :\n{data[column_name].value_counts()}")
        print('-' * 79)


# --------------------------------------------------------------------
# -- BARPLOT DES VARIABLES CATEGORIELLES
# --------------------------------------------------------------------


def plot_categorical_variables_bar(data, column_name, figsize=(18, 6),
                                   percentage_display=True,
                                   plot_defaulter=True, rotation=0,
                                   horizontal_adjust=0,
                                   fontsize_percent='xx-small',
                                   palette1='Set1',
                                   palette2='Set2'):
    '''
    Function to plot Categorical Variables Bar Plots

    Inputs:
        data: DataFrame
            The DataFrame from which to plot
        column_name: str
            Column's name whose distribution is to be plotted
        figsize: tuple, default = (18,6)
            Size of the figure to be plotted
        percentage_display: bool, default = True
            Whether to display the percentages on top of Bars in Bar-Plot
        plot_defaulter: bool
            Whether to plot the Bar Plots for Defaulters or not
        rotation: int, default = 0
            Degree of rotation for x-tick labels
        horizontal_adjust: int, default = 0
            Horizontal adjustment parameter for percentages displayed on the top of Bars of Bar-Plot
        fontsize_percent: str, default = 'xx-small'
            Fontsize for percentage Display

    '''

    print(
        f"Nombre de catégories uniques pour {column_name} = {len(data[column_name].unique())}")

    plt.figure(figsize=figsize, tight_layout=True)
    sns.set(style='whitegrid', font_scale=1.2)

    # plotting overall distribution of category
    plt.subplot(1, 2, 1)
    data_to_plot = data[column_name].value_counts().sort_values(ascending=False)
    ax = sns.barplot(x=data_to_plot.index, y=data_to_plot, palette=palette1)

    if percentage_display:
        total_datapoints = len(data[column_name].dropna())
        for p in ax.patches:
            ax.text(
                p.get_x() +
                horizontal_adjust,
                p.get_height() +
                0.005 *
                total_datapoints,
                '{:1.02f}%'.format(
                    p.get_height() *
                    100 /
                    total_datapoints),
                fontsize=fontsize_percent)

    plt.xlabel(column_name, labelpad=10)
    plt.title('Toutes TARGET', pad=20, fontsize=30)
    plt.xticks(rotation=rotation, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Nombre', fontsize=20)

    # plotting distribution of category for Defaulters
    if plot_defaulter:
        percentage_defaulter_per_category = (data[column_name][data.TARGET == 1].value_counts(
        ) * 100 / data[column_name].value_counts()).dropna().sort_values(ascending=False)

        plt.subplot(1, 2, 2)
        sns.barplot(x=percentage_defaulter_per_category.index,
                    y=percentage_defaulter_per_category, palette=palette2)
        plt.ylabel(
            'Pourcentage par catégorie pour les défaillants',
            fontsize=20)
        plt.xlabel(column_name, labelpad=10)
        plt.xticks(rotation=rotation, fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('Défaillants seuls', pad=20, fontsize=30)

    plt.suptitle(f'Répartition de {column_name}', fontsize=40)
    plt.show()


def plot_categorical_variable_bar(data, column_name, figsize=(18, 6),
                                  percentage_display=True, rotation=0,
                                  horizontal_adjust=0,
                                  fontsize_percent='xx-small',
                                  palette1='Set1'):
    '''
    Function to plot Categorical Variables Bar Plots

    Inputs:
        data: DataFrame
            The DataFrame from which to plot
        column_name: str
            Column's name whose distribution is to be plotted
        figsize: tuple, default = (18,6)
            Size of the figure to be plotted
        percentage_display: bool, default = True
            Whether to display the percentages on top of Bars in Bar-Plot
        plot_defaulter: bool
            Whether to plot the Bar Plots for Defaulters or not
        rotation: int, default = 0
            Degree of rotation for x-tick labels
        horizontal_adjust: int, default = 0
            Horizontal adjustment parameter for percentages displayed on the top of Bars of Bar-Plot
        fontsize_percent: str, default = 'xx-small'
            Fontsize for percentage Display

    '''

    print(
        f"Nombre de catégories uniques pour {column_name} = {len(data[column_name].unique())}")

    plt.figure(figsize=figsize, tight_layout=True)
    sns.set(style='whitegrid', font_scale=1.2)

    data_to_plot = data[column_name].value_counts().sort_values(ascending=False)
    ax = sns.barplot(x=data_to_plot.index, y=data_to_plot, palette=palette1)

    if percentage_display:
        total_datapoints = len(data[column_name].dropna())
        for p in ax.patches:
            ax.text(
                p.get_x() +
                horizontal_adjust,
                p.get_height() +
                0.005 *
                total_datapoints,
                '{:1.02f}%'.format(
                    p.get_height() *
                    100 /
                    total_datapoints),
                fontsize=fontsize_percent)

    plt.xlabel(column_name, labelpad=10)
    plt.title(f'Barplot de {column_name}', pad=20, fontsize=30)
    plt.xticks(rotation=rotation, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Nombre', fontsize=20)

    plt.show()


# --------------------------------------------------------------------
# -- PIEPLOT DES VARIABLES CATEGORIELLES
# --------------------------------------------------------------------


def plot_categorical_variables_pie(
        data,
        column_name,
        plot_defaulter=True,
        hole=0):
    '''
    Function to plot categorical variables Pie Plots
    Inputs:
        data: DataFrame
            The DataFrame from which to plot
        column_name: str
            Column's name whose distribution is to be plotted
        plot_defaulter: bool
            Whether to plot the Pie Plot for Defaulters or not
        hole: int, default = 0
            Radius of hole to be cut out from Pie Chart
    '''
    if plot_defaulter:
        cols = 2
        specs = [[{'type': 'domain'}, {'type': 'domain'}]]
        titles = ['Toutes TARGET', 'Défaillants seuls']
    else:
        cols = 1
        specs = [[{'type': 'domain'}]]
        titles = [f'Répartition de la variable {column_name}']

    values_categorical = data[column_name].value_counts()
    labels_categorical = values_categorical.index

    fig = make_subplots(rows=1, cols=cols,
                        specs=specs,
                        subplot_titles=titles)

    fig.add_trace(
        go.Pie(
            values=values_categorical,
            labels=labels_categorical,
            hole=hole,
            textinfo='percent',
            textposition='inside'),
        row=1,
        col=1)

    if plot_defaulter:
        percentage_defaulter_per_category = data[column_name][data.TARGET == 1].value_counts(
        ) * 100 / data[column_name].value_counts()
        percentage_defaulter_per_category.dropna(inplace=True)
        percentage_defaulter_per_category = percentage_defaulter_per_category.round(
            2)

        fig.add_trace(
            go.Pie(
                values=percentage_defaulter_per_category,
                labels=percentage_defaulter_per_category.index,
                hole=hole,
                textinfo='percent',
                hoverinfo='label+value'),
            row=1,
            col=2)

    fig.update_layout(title=f'Répartition de la variable {column_name}')
    fig.show()


# --------------------------------------------------------------------
# -- AFFICHE DISTPLOT ou CDF ou BOXPLOT ou VIOLINPLOT DES VARIABLES CONTINUES
# --------------------------------------------------------------------


def plot_continuous_variables(data, column_name,
                              plots=['distplot', 'CDF', 'box', 'violin'], 
                              scale_limits=None, figsize=(20, 9),
                              histogram=True, log_scale=False,
                              palette=['SteelBlue', 'Crimson']):
    '''
    Function to plot continuous variables distribution
    Inputs:
        data: DataFrame
            The DataFrame from which to plot.
        column_name: str
            Column's name whose distribution is to be plotted.
        plots: list, default = ['distplot', 'CDF', box', 'violin']
            List of plots to plot for Continuous Variable.
        scale_limits: tuple (left, right), default = None
            To control the limits of values to be plotted in case of outliers.
        figsize: tuple, default = (20,8)
            Size of the figure to be plotted.
        histogram: bool, default = True
            Whether to plot histogram along with distplot or not.
        log_scale: bool, default = False
            Whether to use log-scale for variables with outlying points.
    '''
    data_to_plot = data.copy()
    if scale_limits:
        # taking only the data within the specified limits
        data_to_plot[column_name] = data[column_name][(
            data[column_name] > scale_limits[0]) & (data[column_name] < scale_limits[1])]

    number_of_subplots = len(plots)
    plt.figure(figsize=figsize)
    sns.set_style('whitegrid')

    for i, ele in enumerate(plots):
        plt.subplot(1, number_of_subplots, i + 1)
        plt.subplots_adjust(wspace=0.25)

        if ele == 'CDF':
            # making the percentile DataFrame for both positive and negative
            # Class Labels
            percentile_values_0 = data_to_plot[data_to_plot.TARGET == 0][[
                column_name]].dropna().sort_values(by=column_name)
            percentile_values_0['Percentile'] = [
                ele / (len(percentile_values_0) - 1) for ele in range(len(percentile_values_0))]

            percentile_values_1 = data_to_plot[data_to_plot.TARGET == 1][[
                column_name]].dropna().sort_values(by=column_name)
            percentile_values_1['Percentile'] = [
                ele / (len(percentile_values_1) - 1) for ele in range(len(percentile_values_1))]

            plt.plot(
                percentile_values_0[column_name],
                percentile_values_0['Percentile'],
                color='SteelBlue',
                label='Non-Défaillants')
            plt.plot(
                percentile_values_1[column_name],
                percentile_values_1['Percentile'],
                color='crimson',
                label='Défaillants')
            plt.xlabel(column_name, fontsize=16)
            plt.ylabel('Probabilité', fontsize=16)
            plt.title('CDF de {}'.format(column_name), fontsize=18)
            plt.legend(fontsize='medium')
            if log_scale:
                plt.xscale('log')
                plt.xlabel(column_name + ' - (log-scale)')

        if ele == 'distplot':
            sns.distplot(data_to_plot[column_name][data['TARGET'] == 0].dropna(
            ), label='Non-Défaillants', hist=False, color='SteelBlue')
            sns.distplot(data_to_plot[column_name][data['TARGET'] == 1].dropna(
            ), label='Défaillants', hist=False, color='Crimson')
            plt.xlabel(column_name, fontsize=16)
            plt.ylabel('Probability Density', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(fontsize=18)
            plt.title("Dist-Plot de {}".format(column_name), fontsize=18)
            if log_scale:
                plt.xscale('log')
                plt.xlabel(f'{column_name} (log scale)', fontsize=16)

        if ele == 'violin':
            sns.violinplot(x='TARGET', y=column_name, data=data_to_plot, palette=palette)
            plt.title("Violin-Plot de {}".format(column_name), fontsize=18)
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')

        if ele == 'box':
            sns.boxplot(x='TARGET', y=column_name, data=data_to_plot, palette=palette)
            plt.title("Box-Plot de {}".format(column_name), fontsize=18)
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)', fontsize=16)
            plt.xlabel('TARGET', fontsize=16)
            plt.ylabel(f'{column_name}', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

    plt.show()
    

def plot_continuous_variable(data, column_name,
                             plots=['distplot', 'CDF', 'box', 'violin'], 
                             scale_limits=None, figsize=(20, 9),
                             histogram=True, log_scale=False,
                             palette=['SteelBlue', 'Crimson']):
    '''
    Function to plot continuous variables distribution
    Inputs:
        data: DataFrame
            The DataFrame from which to plot.
        column_name: str
            Column's name whose distribution is to be plotted.
        plots: list, default = ['distplot', 'CDF', box', 'violin']
            List of plots to plot for Continuous Variable.
        scale_limits: tuple (left, right), default = None
            To control the limits of values to be plotted in case of outliers.
        figsize: tuple, default = (20,8)
            Size of the figure to be plotted.
        histogram: bool, default = True
            Whether to plot histogram along with distplot or not.
        log_scale: bool, default = False
            Whether to use log-scale for variables with outlying points.
    '''
    data_to_plot = data.copy()
    if scale_limits:
        # taking only the data within the specified limits
        data_to_plot[column_name] = data[column_name][(
            data[column_name] > scale_limits[0]) & (data[column_name] < scale_limits[1])]

    number_of_subplots = len(plots)
    plt.figure(figsize=figsize)
    sns.set_style('whitegrid')

    for i, ele in enumerate(plots):
        plt.subplot(1, number_of_subplots, i + 1)
        plt.subplots_adjust(wspace=0.25)

        if ele == 'CDF':
            # making the percentile DataFrame for both positive and negative
            # Class Labels
            percentile_values_0 = data_to_plot[[column_name]].dropna().sort_values(by=column_name)
            percentile_values_0['Percentile'] = [
                ele / (len(percentile_values_0) - 1) for ele in range(len(percentile_values_0))]

            plt.plot(
                percentile_values_0[column_name],
                percentile_values_0['Percentile'],
                color='SteelBlue')
            plt.xlabel(column_name, fontsize=16)
            plt.ylabel('Probabilité', fontsize=16)
            plt.title('CDF de {}'.format(column_name), fontsize=18)
            plt.legend(fontsize='medium')
            if log_scale:
                plt.xscale('log')
                plt.xlabel(column_name + ' - (log-scale)')

        if ele == 'distplot':
            sns.distplot(data_to_plot[column_name].dropna(), hist=False,
                         color='SteelBlue')
            plt.xlabel(column_name, fontsize=16)
            plt.ylabel('Probability Density', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(fontsize=18)
            plt.title("Dist-Plot de {}".format(column_name), fontsize=18)
            if log_scale:
                plt.xscale('log')
                plt.xlabel(f'{column_name} (log scale)', fontsize=16)

        if ele == 'violin':
            sns.violinplot(x=column_name, data=data_to_plot)
            plt.title("Violin-Plot de {}".format(column_name), fontsize=18)
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')

        if ele == 'box':
            sns.boxplot(x=column_name, data=data_to_plot, palette=palette)
            plt.title("Box-Plot de {}".format(column_name), fontsize=18)
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)', fontsize=16)
            plt.xlabel(f'{column_name}', fontsize=16)
            plt.xticks(fontsize=16)

    plt.show()

    
# --------------------------------------------------------------------
# -- AFFICHE LES QUANTILES POUR LA VARIABLE
# --------------------------------------------------------------------


def print_percentiles(data, column_name, percentiles=None):
    '''
    Function to print percentile values for given column
    Inputs:
        data: DataFrame
            The DataFrame from which to print percentiles
        column_name: str
            Column's name whose percentiles are to be printed
        percentiles: list, default = None
            The list of percentiles to print, if not given, default are printed
    '''
    print('-' * 79)
    print(f'Pecentiles de la variable {column_name}')
    if not percentiles:
        percentiles = list(range(0, 80, 25)) + list(range(90, 101, 2))
    for i in percentiles:
        
        print(
            f'Pecentile {i} = {np.percentile(data[column_name].dropna(), i)}')
    print("-" * 79)
    
    
# --------------------------------------------------------------------
# --
# --------------------------------------------------------------------


def plot_cdf(data, column_name, log_scale=False, figsize=(12, 8)):
    '''
    Function to plot CDF of a continuour variable
    Inputs:
        data: DataFrame
            The DataFrame from which to plot
        column_name: str
            Column's name whose CDF is to be plotted
        log_scale: bool, default = True
            Whether to use log-scale (for widely varying values) or not
        figsize: tuple, default = (12,8)
            The size of figure to be plotted
    '''
    percentile_values = data[[column_name]
                             ].dropna().sort_values(by=column_name)
    percentile_values['Percentile'] = [
        ele / (len(percentile_values) - 1) for ele in range(len(percentile_values))]

    plt.figure(figsize=figsize)
    if log_scale:
        plt.xscale('log')
        plt.xlabel(column_name + ' - (log-scale)')
    else:
        plt.xlabel(column_name)
    plt.plot(
        percentile_values[column_name],
        percentile_values['Percentile'],
        color='red')
    plt.ylabel('Probability')
    plt.title('CDF of {}'.format(column_name))
    plt.show()


def plot_barplot_comp_target(dataframe, feature_name,
                             labels=['Non-défaillant', 'Défaillant'],
                             palette=['SteelBlue', 'crimson'],
                             rotation=0):
    '''
    Barplot de comparaison des catégories par target.
    Parameters
    ----------
    dataframe : dataframe, obligatoire.
    feature_name : variable, obligatoire.
    Returns
    -------
    None.
    '''
    sns.countplot(x=dataframe[feature_name], hue=dataframe.TARGET,
                  data=dataframe, palette=palette)
    plt.xticks(rotation=rotation)
    plt.title(f'Distribution de {feature_name} par défaillant/non-défaillant')
    plt.legend(labels=labels,
               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()