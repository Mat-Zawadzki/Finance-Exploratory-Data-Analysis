import numpy as np
from typing import Literal

class OutlierTransform:
    """Transforms the outliers in the dataset.

    Attributes:
        df = DataFrame to apply the transformations to
        df_columns: List of the DataFrame columns
    """
   
    def __init__(self, data):
        self.df = data
        self.df_columns = data.columns

    def replace_by_z_score(self, df_column: str, threshold: int = 3):
        """
        Removes outliers with the z-score method of removal
        z-score is better when the data is not heavily skewed

        Args:
            df_column (str): column to apply the z-score removal to
            impute_val (str, optional): 
                Value to impute the data with mean or median. Defaults to "mean".
            threshold (int, optional): z-score threshold to set. Defaults to 3.

        Returns:
            _type_: _description_
        """ 
        outliers = []
        mean = self.df[df_column].mean()
        std = self.df[df_column].std()
        outliers = self.df.loc[((self.df[df_column] - mean) / std) > threshold]
        self.df[df_column] = self.df.loc[((self.df[df_column] - mean) / std) > threshold] = mean
        return outliers

    def replace_using_IQR(self, df_column):
        """
        Removes outliers with the IQR method of removal 
        IQR is better for heavily skewed data

        Args:
            df_column (str): Name of the column to remove outliers from
            impute_option (str): Method selected to impute the data with.
        """
        median = self.df[df_column].median()
        data = sorted(self.df[df_column])
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        IQR = q3-q1
        lower_bound = q1-(1.5*IQR)
        upper_bound = q3+(1.5*IQR)
        outliers = self.df.loc[(self.df[df_column] > upper_bound) | (self.df[df_column] < lower_bound), df_column]
        self.df \
            .loc[(self.df[df_column] > upper_bound) | (self.df[df_column] < lower_bound), df_column] \
            = median
        return outliers

    def replace_outliers(self, method: Literal["IQR", "z_scores"] = "IQR"):
        """
        Applies the outlier transformation based on the
        selected method using the selected imputation value

        Args:
            impute_option (str Literal): Wether to use the mean or median to replace outliers with.
            method: Method to replace the outliers with.
        """
        for column in self.df.columns:
            if method == "IQR":
                outliers = self.replace_using_IQR(column)
                return outliers
            elif method == "z_scores":
                outliers = self.replace_by_z_score(column)
                return outliers
            else:
                print("Invalid replacement option")
            
