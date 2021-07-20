""" Librairie personnelle pour visualisations des graphiques seaborn,
    matplotlib, heatmap...
"""

#! /usr/bin/env python3
# coding: utf-8

# ====================================================================
# Outil visualisation -  projet 3 Openclassrooms
# Version : 0.0.1 - CRE LR 13/03/2021
# Version : 0.0.2 - CRE LR 28/04/2021 Ajout pour P5 Segmentation clientèle
# ====================================================================
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import missingno
import seaborn as sns
from IPython.display import display
from wordcloud import WordCloud
from statsmodels.graphics.gofplots import qqplot
import outils_data
import texthero as hero

# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.2'


# --------------------------------------------------------------------
# -- BOITE A MOUSTACHE 3 graphiques par colonnes avec zoom et limites
# --------------------------------------------------------------------


def trace_boite_moustache(dataframe, variable, titre,
                          y_zoom_sup_bas, y_zoom_sup_haut,
                          y_zoom_inf_bas, y_zoom_inf_haut, unite,
                          limite_basse, limite_haute):
    """
    Boite à moustache des variables qualitatives avec zoom supérieur et inférieur
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : colonne dont on veut voir les outliers
                titre : titre du graphique (str)
                y_zoom_sup_bas : limite du zoom supérieur bas(int)
                y_zoom_sup_haut : limite du zoom supérieur haut(int)
                y_zoom_inf_bas : limite du zoom inférieur bas(int)
                y_zoom_inf_haut : limite du zoom inférieur haut(int)
                unite : g ou kcal ou kJ (str)
                limite_basse : trace un trait de la limite basse à ne pas dépasser (int)
                limite_haute : trace un trait de la limite haute à ne pas dépasser (int)
    @param OUT :None
    """
    # Visualisation des outliers
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.boxplot(data=dataframe[variable], color='SteelBlue')
    plt.title(titre)
    # plt.ylim([0,200])
    plt.ylabel(unite + '/100g ou 100ml')
    plt.xlabel = ''
    plt.axhline(y=limite_haute, color='r')
    plt.axhline(y=limite_basse, color='r')

    plt.subplot(1, 3, 2)
    sns.boxplot(data=dataframe[variable], color='SteelBlue')
    plt.title(titre + ' zoom')
    plt.ylim([y_zoom_sup_bas, y_zoom_sup_haut])
    plt.ylabel(unite + '/100g ou 100ml')
    plt.xlabel = ''
    plt.axhline(y=limite_haute, color='r')
    plt.axhline(y=limite_basse, color='r')
    plt.axhline(y=0, color='r')

    plt.subplot(1, 3, 3)
    sns.boxplot(data=dataframe[variable], color='SteelBlue')
    plt.title(titre + ' zoom')
    plt.ylim([y_zoom_inf_bas, y_zoom_inf_haut])
    plt.ylabel(unite + '/100g ou 100ml')
    plt.xlabel = ''
    plt.axhline(y=limite_haute, color='r')
    plt.axhline(y=limite_basse, color='r')
    plt.show()

# --------------------------------------------------------------------
# -- HISTPLOT 2 graphiques avec zoom
# --------------------------------------------------------------------


def trace_histplot_gen_zoom(
        dataframe,
        variable,
        titre,
        xlabel,
        xlim_bas,
        xlim_haut,
        ylim_bas,
        ylim_haut,
        kde=True,
        mean_median_mode=True,
        mean_median_zoom=False):
    """
    Histplot pour les variables quantitatives général + histplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : colonne dont on veut voir les histplot
                titre : titre du graphique (str)
                xlabel:légende des abscisses
                xlim_bas : limite du zoom supérieur bas(int)
                xlim_haut : limite du zoom supérieur haut(int)
                ylim_bas : limite du zoom inférieur bas(int)
                ylim_haut : limite du zoom inférieur haut(int)
                kde : boolean pour tracer la distribution normale
                mean_median_mode : boolean pour tracer la moyenne, médiane et mode
                mean_median_zoom : boolean pour tracer la moyenne et médiane sur le graphique zoomé
    @param OUT :None
    """
    # Distplot général + zoom
    fig = plt.figure(figsize=(15, 6))

    data = dataframe[variable]
    ax1 = fig.add_subplot(1, 2, 1)
    ax1 = sns.histplot(data, kde=kde, color='SteelBlue')
    if mean_median_mode:
        ax1.vlines(data.mean(), *ax1.get_ylim(), color='red', ls='-', lw=1.5)
        ax1.vlines(
            data.median(),
            *ax1.get_ylim(),
            color='green',
            ls='-.',
            lw=1.5)
        ax1.vlines(
            data.mode()[0],
            *ax1.get_ylim(),
            color='goldenrod',
            ls='--',
            lw=1.5)
    ax1.legend(['mode', 'mean', 'median'])
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel('Nombre', fontsize=12)
    ax1.set_title(titre, fontsize=14)
    plt.grid(False)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2 = sns.histplot(dataframe[variable], kde=kde, color='SteelBlue')
    if mean_median_zoom:
        ax2.vlines(data.mean(), *ax2.get_ylim(), color='red', ls='-', lw=1.5)
        ax2.vlines(
            data.median(),
            *ax2.get_ylim(),
            color='green',
            ls='-.',
            lw=1.5)
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel('Nombre', fontsize=12)
    ax2.set_xlim(xlim_bas, xlim_haut)
    ax2.set_ylim(ylim_bas, ylim_haut)
    ax2.set_title(titre + ' Zoom', fontsize=14)
    plt.grid(False)

    plt.show()

# --------------------------------------------------------------------
# -- HISTPLOT BOXPLOT QQPLOT
# --------------------------------------------------------------------


def trace_countplot(dataframe, liste_variables):
    """
    Suivi des dipsersions : boxplot et qqplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                liste_variables : colonnes dont on veut voir les outliers
    @param OUT :None
    """
    for col in liste_variables:
        plt.figure(figsize=(8, 6))
        sns.set_style("white")
        sns.countplot(y=col, data=dataframe, color='SteelBlue')
        plt.title('Distribution de ' + col)
        plt.show()

# --------------------------------------------------------------------
# -- HISTPLOT BOXPLOT QQPLOT
# --------------------------------------------------------------------


def trace_multi_histplot_boxplot_qqplot(dataframe, liste_variables):
    """
    Suivi des dipsersions : boxplot et qqplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                liste_variables : colonnes dont on veut voir les outliers
    @param OUT :None
    """
    for col in liste_variables:
        trace_histplot_boxplot_qqplot(dataframe, col)


def trace_histplot_boxplot_qqplot(dataframe, variable):
    """
    Suivi des dipsersions : boxplot et qqplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : colonne dont on veut voir les outliers
    @param OUT :None
    """
    # Boxplot + qqplot
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle('Distribution de ' + variable, fontsize=16)

    data = dataframe[variable]

    ax0 = fig.add_subplot(1, 3, 1)
    sns.histplot(data, kde=True, ax=ax0)
    plt.xticks(rotation=60)
    plt.grid(False)

    ax1 = fig.add_subplot(1, 3, 2)
    sns.boxplot(data=data, color='SteelBlue', ax=ax1)
    plt.grid(False)

    ax2 = fig.add_subplot(1, 3, 3)
    qqplot(data,
           line='r',
           **{'markersize': 5,
              'mec': 'k',
              'color': 'SteelBlue'},
           ax=ax2)
    plt.grid(False)

    plt.show()

# --------------------------------------------------------------------
# -- BOXPLOT QQPLOT
# --------------------------------------------------------------------


def trace_dispersion_boxplot_qqplot(dataframe, variable, titre, unite):
    """
    Suivi des dipsersions : boxplot et qqplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : colonne dont on veut voir les outliers
                titre :titre pour les graphiques (str)
                unite : unité pour ylabel boxplot (str)
    @param OUT :None
    """
    # Boxplot + qqplot
    fig = plt.figure(figsize=(15, 6))

    data = dataframe[variable]

    ax1 = fig.add_subplot(1, 2, 1)
    box = sns.boxplot(data=data, color='SteelBlue', ax=ax1)
    box.set(ylabel=unite)

    plt.grid(False)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2 = qqplot(data,
                 line='r',
                 **{'markersize': 5,
                    'mec': 'k',
                    'color': 'SteelBlue'},
                 ax=ax2)
    plt.grid(False)

    fig.suptitle(titre, fontweight='bold', size=14)
    plt.show()

# --------------------------------------------------------------------
# -- PIEPLOT
# --------------------------------------------------------------------


def trace_pieplot(dataframe, variable, titre, legende, liste_colors):
    """
    Suivi des dipsersions : bosplot et qqplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : colonne dont on veut voir les outliers (str)
                titre :titre pour les graphiques (str)
                legende : titre de la légende
                liste_colors : liste des couleurs
    @param OUT :None
    """

    plt.figure(figsize=(30, 7))
    plt.title(titre, size=16)
    nb_par_var = dataframe[variable].sort_values().value_counts()
    # nb_par_var = nb_par_var.loc[sorted(nb_par_var.index)]
    explode = [0.1]
    for i in range(len(nb_par_var) - 1):
        explode.append(0)
    wedges, texts, autotexts = plt.pie(
        nb_par_var, labels=nb_par_var.index, autopct='%1.1f%%', colors=liste_colors, textprops={
            'fontsize': 16, 'color': 'black', 'backgroundcolor': 'w'}, explode=explode)
    axes = plt.gca()
    axes.legend(
        wedges,
        nb_par_var.index,
        title=legende,
        borderaxespad=0.,
        loc=2,
        fontsize=14,
        bbox_to_anchor=(1.5, 1))
    plt.show()


# --------------------------------------------------------------------
# -- WORDCLOUD + TABLEAU DE FREQUENCE
# --------------------------------------------------------------------


def affiche_wordcloud_tabfreq(
        dataframe,
        variable,
        nom,
        affword=True,
        affgraph=True,
        afftabfreq=True,
        nb_lignes=10):
    """
    Affiche les 'noms' les plus fréquents (wordcloud) et le tableau de fréquence (10 1ères lignes)
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : variable dont on veut voir la fréquence obligatoire
                nom : text affiché dans le tableau de fréquence obligatoire
                nb_lignes : nombre de lignes affichées dans le tab des fréquences facultatives
                affword : booléen : affiche le wordcloud ?
                affgraph : booléen : affiche le graphique de répartition en pourcentage ?
                afftabfreq : booléen : affiche le tableau des fréquences ?
    @param OUT : None
    """
    # Préparation des variables de travail
    dico = dataframe.groupby(variable)[variable].count(
    ).sort_values(ascending=False).to_dict()
    col1 = 'Nom_' + nom
    col2 = 'Nbr_' + nom
    col3 = 'Fréquence (%)'
    df_gpe = pd.DataFrame(dico.items(), columns=[col1, col2])
    df_gpe[col3] = (df_gpe[col2] * 100) / len(dataframe)
    df_gp_red = df_gpe.head(nb_lignes)

    if affword:
        # affiche le wordcloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=100).generate_from_frequencies(dico)
        plt.figure(figsize=(12, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    if affgraph:
        # Barplot de répartition
        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 4))
        sns.barplot(
            y=df_gp_red[col1],
            x=df_gp_red[col3],
            data=df_gp_red,
            color='SteelBlue')
        plt.title('Répartition du nombre de ' + nom)
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    if afftabfreq:
        # affiche le tableau des fréquences
        display(df_gp_red.style.hide_index())


# --------------------------------------------------------------------
# -- WORDCLOUD + multiple variables
# --------------------------------------------------------------------


def affiche_wordcloud_hue(dataframe, variable, var_hue, nb_mots):

    liste_hue = dataframe[var_hue].unique().tolist()
    for cat in liste_hue:
        print('Mots les plus fréquents de la catégorie : ' + cat)
        hero.wordcloud(dataframe[dataframe[var_hue] == cat][variable],
                       max_words=nb_mots)
        plt.show()


# ---------------------------------------------------------------------------
# -- EDA DES TIME SERIES
# ---------------------------------------------------------------------------


def time_series_plot(df_work):
    """Given dataframe, generate times series plot of numeric data by daily,
       monthly and yearly frequency"""
    print("\nTo check time series of numeric data  by daily, monthly and yearly frequency")
    if len(df_work.select_dtypes(include='datetime64').columns) > 0:
        for col in df_work.select_dtypes(include='datetime64').columns:
            for plotting in ['D', 'M', 'Y']:
                if plotting == 'D':
                    print("Plotting daily data")
                elif plotting == 'M':
                    print("Plotting monthly data")
                else:
                    print("Plotting yearly data")
                for col_num in df_work.select_dtypes(
                        include=np.number).columns:
                    __ = df_work.copy()
                    __ = __.set_index(col)
                    transp = __.resample(plotting).sum()
                    axes = transp[[col_num]].plot()
                    axes.set_ylim(bottom=0)
                    axes.get_yaxis().set_major_formatter(
                        matplotlib.ticker.FuncFormatter(
                            lambda x, p: format(
                                int(x), ',')))
                    plt.show()

# --------------------------------------------------------------------------
# -- EDA DES VARIABLES QUANTITATIVES
# --------------------------------------------------------------------------

# Génère EDA pour les variables quantitatives du dataframe transmis


def numeric_eda(df_work, hue=None):
    """Génère EDA pour les variables quantitatives du dataframe transmis
       @param in : df_work dataframe obligatoire
                   hue non obigatoire
       @param out : none
    """
    print("----------------------------------------------------")
    print("\nEDA variables quantitatives : \nDistribution des variables quantitatives\n")
    print(df_work.describe().T)
    columns = df_work.select_dtypes(include=np.number).columns
    figure = plt.figure(figsize=(20, 10))
    figure.add_subplot(1, len(columns), 1)
    for index, col in enumerate(columns):
        if index > 0:
            figure.add_subplot(1, len(columns), index + 1)
        sns.boxplot(y=col, data=df_work, boxprops={'facecolor': 'None'})
    figure.tight_layout()
    plt.show()

    if len(df_work.select_dtypes(include='category').columns) > 0:
        for col_num in df_work.select_dtypes(include=np.number).columns:
            for col in df_work.select_dtypes(include='category').columns:
                fig = sns.catplot(
                    x=col,
                    y=col_num,
                    kind='violin',
                    data=df_work,
                    height=5,
                    aspect=2)
                fig.set_xticklabels(rotation=90)
                plt.show()

    # Affiche le pairwise joint distributions
    print("\nAffiche pairplot des variables quantitatives")
    if hue is None:
        sns.pairplot(df_work.select_dtypes(include=np.number))
    else:
        sns.pairplot(df_work.select_dtypes(
            include=np.number).join(df_work[[hue]]), hue=hue)
    plt.show()

# --------------------------------------------------------------------------
# -- EDA DES VARIABLES QUALITATIVES
# --------------------------------------------------------------------------

# Top 5 des modalités uniques par variable qualitative


def top5(df_work):
    """Affiche le top 5 des modalités uniques par variables qualitatives
       @param in : df_work dataframe obligatoire
       @param out : none
    """
    print("----------------------------------------------------")
    columns = df_work.select_dtypes(include=['object', 'category']).columns
    for col in columns:
        print("Top 5 des modalités uniques de : " + col)
        print(df_work[col].value_counts().reset_index()
              .rename(columns={"index": col, col: "Count"})[
              :min(5, len(df_work[col].value_counts()))])
        print(" ")


# Génère EDA pour les variables qualitatives du dataframe transmis
def categorical_eda(df_work, hue=None):
    """Génère EDA pour les variables qualitatives du dataframe transmis
       @param in : df_work dataframe obligatoire
                   hue non obigatoire
       @param out : none
    """
    print("----------------------------------------------------")
    print("\nEDA variables qualitatives : \nDistribution des variables qualitatives")
    print(df_work.select_dtypes(include=['object', 'category']).nunique())
    top5(df_work)
    # Affiche count distribution des variables qualitatives
    for col in df_work.select_dtypes(include='category').columns:
        fig = sns.catplot(x=col, kind="count", data=df_work, hue=hue)
        fig.set_xticklabels(rotation=90)
        plt.show()

# ---------------------------------------------------------------------------
# -- EDA DE TOUTES LES VARIABLES : QUANTITATIVES, QUALITATIVES
# ---------------------------------------------------------------------------


def eda(df_work):
    """Génère l'analyse exploratoire du dataframe transmis pour toutes les variables"""

    print("----------------------------------------------------")

    # Controle que le paramètre transmis est un dataframe pandas
    # if type(df_work) != pd.core.frame.DataFrame:
    if isinstance(df_work, pd.core.frame.DataFrame):
        raise TypeError("Seul un dataframe pandas est autorisé en entrée")

    # Remplace les données avec vide ou espace par NaN
    df_work = df_work.replace(r'^\s*$', np.nan, regex=True)

    print("----------------------------------------------------")
    print("3 premières lignes du jeu de données:")
    print(df_work.head(3))

    print("----------------------------------------------------")
    print("\nEDA des variables: \n (1) Total du nombre de données \n  \
          (2) Types ds colonnes \n (3) Any null values\n")
    print(df_work.info())

    # Affichage des valeurs manquantes
    if df_work.isnull().any(axis=None):
        print("----------------------------------------------------")
        print("\nPrévisualisation des données avec valeurs manquantes:")
        print(df_work[df_work.isnull().any(axis=1)].head(3))
        missingno.matrix(df_work)
        plt.show()

    outils_data.get_missing_values(df_work, True, True)

    print("----------------------------------------------------")
    # Statitstique du nombre de données dupliquées
    if len(df_work[df_work.duplicated()]) > 0:
        print("\n***Nombre de données dupliquées : ",
              len(df_work[df_work.duplicated()]))
        print(df_work[df_work.duplicated(keep=False)].sort_values(
            by=list(df_work.columns)).head())
    else:
        print("\nAucune donnée dupliquée trouvée")

    # EDA des variables qualitatives
    print("----------------------------------------------------")
    print('-- EDA DES VARIABLES QUALITATIVES')
    categorical_eda(df_work)

    # EDA des variables quantitatives
    print("----------------------------------------------------")
    print('-- EDA DES VARIABLES QUANTITATIVES')
    numeric_eda(df_work)

    # Affiche les Plot time series plot des variables quantitatives
    time_series_plot(df_work)

# ---------------------------------------------------------------------------
# -- KDEPLOT Graph densité pour 1 ou plusieurs colonne d'un dataframe
# ---------------------------------------------------------------------------


def plot_graph(df_work):
    """Graph densité pour 1 ou plusieurs colonne d'un dataframe
       @param in : df_work dataframe obligatoire
       @param out : none
    """

    plt.figure(figsize=(10, 5))
    axes = plt.axes()

    label_patches = []
    colors = ['Blue', 'SeaGreen', 'Sienna', 'DodgerBlue', 'Purple']

    i = 0
    for col in df_work.columns:
        label = col
        sns.kdeplot(df_work[col], color=colors[i])
        label_patch = mpatches.Patch(
            color=colors[i],
            label=label)
        label_patches.append(label_patch)
        i += 1
    plt.xlabel('')
    plt.legend(
        handles=label_patches,
        bbox_to_anchor=(
            1.05,
            1),
        loc=2,
        borderaxespad=0.,
        facecolor='white')
    plt.grid(False)
    axes.set_facecolor('white')

    plt.show()


# --------------------------------------------------------------------
# -- HEATMAP BIEN LISIBLE
# --------------------------------------------------------------------
# https://www.kaggle.com/drazen/heatmap-with-sized-markers


def heatmap(x_param, y_param, **kwargs):
    """
    Heatmap personnalisée
    Parameters
    ----------
    x_param : TYPE
        DESCRIPTION.
    y_param : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.
    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1] * len(x_param)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256  # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors)

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        # Range of values that will be mapped to the palette, i.e. min and max
        # possible correlation
        color_min, color_max = min(color), max(color)

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        # position of value in the input range, relative to the length of
        # the input range
        val_position = float((val - color_min)) / (color_max - color_min)
        # bound the position betwen 0 and 1
        val_position = min(max(val_position, 0), 1)
        # target index in the color palette
        ind = int(val_position * (n_colors - 1))
        return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1] * len(x_param)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        # position of value in the input range, relative to the length of
        # the input range
        val_position = (val - size_min) * 0.99 / \
            (size_max - size_min) + 0.01
        # bound the position betwen 0 and 1
        val_position = min(max(val_position, 0), 1)
        return val_position * size_scale
    if 'x_order' in kwargs:
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x_param]))]
    x_to_num = {p[1]: p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs:
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y_param]))]
    y_to_num = {p[1]: p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(
        1, 15, hspace=0.2, wspace=0.1)  # Setup a 1x10 grid
    # Use the left 14/15ths of the grid for the main plot
    axes = plt.subplot(plot_grid[:, :-1])

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {
        k: v for k,
        v in kwargs.items() if k not in [
            'color',
            'palette',
            'color_range',
            'size',
            'size_range',
            'size_scale',
            'marker',
            'x_order',
            'y_order']}

    axes.scatter(
        x=[x_to_num[v] for v in x_param],
        y=[y_to_num[v] for v in y_param],
        marker=marker,
        s=[value_to_size(v) for v in size],
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    axes.set_xticks([v for k, v in x_to_num.items()])
    axes.set_xticklabels([k for k in x_to_num], rotation=45,
                         horizontalalignment='right')
    axes.set_yticks([v for k, v in y_to_num.items()])
    axes.set_yticklabels([k for k in y_to_num])

    axes.grid(False, 'major')
    axes.grid(True, 'minor')
    axes.set_xticks([t + 0.5 for t in axes.get_xticks()], minor=True)
    axes.set_yticks([t + 0.5 for t in axes.get_yticks()], minor=True)

    axes.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    axes.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    axes.set_facecolor('#F1F1F1')

    # Add color legend on the right side of the plot
    if color_min < color_max:
        # Use the rightmost column of the plot
        axes = plt.subplot(plot_grid[:, -1])

        col_x = [0] * len(palette)  # Fixed x coordinate for the bars
        # y coordinates for each of the n_colors bars
        bar_y = np.linspace(color_min, color_max, n_colors)

        bar_height = bar_y[1] - bar_y[0]
        axes.barh(
            y=bar_y,
            width=[5] * len(palette),  # Make bars 5 units wide
            left=col_x,  # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        # Bars are going from 0 to 5, so lets crop the plot somewhere in the
        # middle
        axes.set_xlim(1, 2)
        axes.grid(False)  # Hide grid
        axes.set_facecolor('white')  # Make background white
        axes.set_xticks([])  # Remove horizontal ticks
        # Show vertical ticks for min, middle and max
        axes.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))
        axes.yaxis.tick_right()  # Show vertical ticks on the right


def corrplot(data, size_scale=500, marker='s'):
    """
    Corrplot
    Parameters
    ----------
    data : dataframe, obligatoire.
    size_scale : taille de l'échelle', optionnel (default : 500).
    marker : type de représentation du marqueur, optionnel (default : 's').
    Returns
    -------
    None.
    """
    corr = pd.melt(data.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        # palette=sns.diverging_palette(20, 220, n=256),
        palette=sns.diverging_palette(240, 10, n=9),
        size=corr['value'].abs(), size_range=[0, 1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale
    )

# --------------------------------------------------------------------
# -- HEATMAP POUR N'AFFICHER QUE LES CORRELATIONS LINEAIRES > une valeur
# --------------------------------------------------------------------


def corrplot_restreint(dataframe, cols_num, valeur_corr_lineaire):
    '''
    Affiche la heatmap en ne gardant que les corrélations linéaires supérieures
    à la valeur transmise.
    Parameters
    ----------
    dataframe : dataframe, obligatoire.
    cols_num : liste des variables numériques
    valeur_corr_lineaire : trace toutes les corrélations linéaires >
                           cette valeur numérique, obligatoire.
    Returns
    -------
    None.
    '''
    data = dataframe[cols_num].corr()
    data_to_plot = data[data > 0.7].fillna(0)
    corrplot(data_to_plot)
