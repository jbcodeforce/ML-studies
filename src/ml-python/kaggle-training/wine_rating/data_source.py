
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def get_statistics(df):
    """
    Function to calculate descriptive statistics, skewness, and kurtosis of a dataframe

    Parameters
    ----------
    df: Dataframe

    Returns
    ----------
    full_stats: dataframe
        Dataframe of descriptive statistics + skewness + kurtosis numeric variables
    """
    desc_stats = df.describe()
    skewness_df = pd.DataFrame(df.skew(numeric_only=True), columns=['skew']).T
    kurtosis_df = pd.DataFrame(df.kurtosis(numeric_only=True), columns=['kurtosis']).T
    full_stats = pd.concat([desc_stats, skewness_df, kurtosis_df])
    print('-------------------------')
    print('Descriptive Statistics')
    print('-------------------------')
    with pd.option_context('display.max_columns', None, 'display.max_rows', None, 'display.expand_frame_repr', False):
        print(full_stats)
        print()

def separate_data(df, response):
    """
    Function to separate features and response variable

    Parameters
    ----------
    df: dataframe
        Dataframe containing true labels of groups (clusters)

    response: string
        String of column name of response variable
    Returns
    ----------
    X: df
        Dataframe containing features from dataset

    y: array-like, (n_samples,)
        Array containing the true labels for each data point
    """
    X = df.drop(response, axis=1)
    y = df[response]
    
    return(X, y)


def scale_data(df):
    """
    Function to scale numerical data

    Parameters
    ----------
    df: dataframe
        Dataframe containing true labels of groups (clusters)

    Returns
    ----------
    df_scaled: dataframe
        Dataframe containing scaled values of numeric variables
    """
    numeric_columns = df.select_dtypes(include=['float64', 'int']).columns
    categorical_columns = df.select_dtypes(exclude=['float64', 'int']).columns
    ct = ColumnTransformer([
        ('scale', StandardScaler(), numeric_columns)
    ], remainder='passthrough')

    # Fit and transform the data
    df_scaled_array = ct.fit_transform(df)
    
    # ColumnTransformer returns an array, convert it back to a DataFrame
    # Combine the column names for transformed and non-transformed columns
    all_columns = numeric_columns.tolist() + categorical_columns.tolist()
    df_scaled = pd.DataFrame(df_scaled_array, columns=all_columns, index=df.index)
    
    return df_scaled