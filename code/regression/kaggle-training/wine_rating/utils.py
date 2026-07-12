
import matplotlib.pyplot as plt
import seaborn as sns

def gen_histograms(df):
    """
    Function to plot histogram of numeric features

    Parameters
    ----------
    df: dataframe
        Dataframe containing the features

    Returns
    ----------
    None:
        Histograms of numeric features
    """
    df.hist(figsize=(15, 10), bins=20)

    plt.show()


def gen_histograms_by_category(df, categorical_column):
    """
    Function to plot histogram of numeric features grouped by 'category'

    Parameters
    ----------
    df: dataframe
        Dataframe containing the features

    category: string
        String containing the name of categorical variable

    Returns
    ----------
    None:
        Histograms of numeric features grouped by category
    """
    numeric_columns = df.select_dtypes(include=['float64']).columns

    # Loop through numeric variables, plot against variety
    for variable in numeric_columns:
        plt.figure(figsize=(8, 4))
        ax = sns.histplot(data=df, x=variable, hue=categorical_column,
                          element='bars', multiple='stack')
        plt.xlabel(f'{variable.capitalize()}')
        plt.title(f'Distribution of {variable.capitalize()}'
                  f' grouped by {categorical_column.capitalize()}')

        legend = ax.get_legend()
        legend.set_title(categorical_column.capitalize())

        plt.show()


def gen_violin_by_category(df, categorical_column):
    """
    Function to generate violin plots of numeric features grouped by 'category'

    Parameters
    ----------
    df: dataframe
        Dataframe containing the features

    category: string
        Name of column in the dataframe

    Returns
    ----------
    None:
        Violin plots of numeric features
    """
    numeric_columns = df.select_dtypes(include=['float64']).columns

    for variable in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.violinplot(x=categorical_column, y=variable, hue=categorical_column, data=df)
        plt.xlabel('Variety')
        plt.ylabel(f'{variable.capitalize()}')
        plt.title(f'Violin plots of {variable.capitalize()} by {categorical_column.capitalize()}')
        plt.show()


def gen_corr_matrix_hmap(df):
    """
    Function to generate correlation matrix of numeric features displayed as heatmap

    Parameters
    ----------
    df: dataframe
        Dataframe containing the features

    Returns
    ----------
    None:
        Heatmap of correlation matrix
    """
    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))

    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})

    plt.title('Correlation Matrix Heat Map')
    plt.show()