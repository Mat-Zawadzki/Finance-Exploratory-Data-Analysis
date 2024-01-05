import pandas as pd


class DataTransform:

    @staticmethod
    def convert_dates(df, date_column): # To datetime to format month-year
        df[date_column] = pd.to_datetime(df[date_column], format="%b-%Y")
        return df

    @staticmethod
    def extract_months(df, column): # Removes "months" from end and changes to float
        df[column] = df[column].str \
            .extract('(\d+)', expand=False).astype(float)
        return df

    @staticmethod
    def convert_to_categorical(df, column):
        df[column] = pd.Categorical(df[column]).codes
        return df
    
    @staticmethod 
    def convert_category_to_numerical(df, column):
        df[column] = df[column].factorize()[0]
        df[column] = df[column] + 1
        return df