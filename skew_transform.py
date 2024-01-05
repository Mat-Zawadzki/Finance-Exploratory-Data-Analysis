from pandas import DataFrame
import numpy as np
import scipy.stats as stats
import pandas as pd

class SkewTransform: 
    # TODO add Yeo Johnson transform
    """Performs the transformation of the skewed columns in the dataframe

       Attributes:
           data (DataFrame): 
                Dataframe which contains all numerical columns in the DataFrame.
           skew_df (DataFrame): 
                DataFrame containing all the skew of all columns.
           thresh (int): 
                Integer value to determine the threshold for the skew. Defaults to 1.
           manual_mapping (dict): 
                Dictionary containing columns as keys and skew method to apply manually to columns.
    """

    def __init__(self, data: DataFrame, skew_df, thresh: int = 1, manual_mapping: dict = {}):
        """
        Initialises the Skew transformer class to perform the skew
        Transformations on skewed columns.
        """
        self.data = data.select_dtypes(include=np.number)
        self.thresh = thresh
        self.skew_df = skew_df.reset_index()
        self.manual_mapping = manual_mapping

    def perform_manual_skew(self):
        """
        Performs skew transformation on columns specified with the manual mapping dictionary.

        Returns:
            skew_change_info (list): 
                A list of strings detailing the changes applied to the individual columns.
        """
        skew_change_info = []
        if self.manual_mapping is None:
            return
        else:
            for key, value in self.manual_mapping.items():
                previous_col_skew = round(self.data[key].skew(), 2)
                if value == 'log':
                    self.data[key] = self.data[key].apply(self.log_transform)
                elif value == 'sqrt':
                    self.data[key] = self.data[key].apply(self.sqrt_transform)
                elif value == 'box_cox':
                    self.data[key] = self.data[key].apply(self.box_cox_transform)
                else:
                    self.data[key] = self.data[key].apply(self.cube_transform)
                skew_change = round(self.data[key].skew(), 2)
                # record the change in the skew value
                skew_changes = [previous_col_skew, skew_change]
                skew_change_info.append(self.build_skew_string(skew_changes, key, value.upper()))
        return skew_change_info

    def get_skewed_columns(self):
        """
        Returns the skewed columns without the columns to be manually changed.

        Returns:
            skewed_columns (list): Returns a list of skewed columns to apply automated transformation to
        """
        skewed_columns = self.skew_df.loc[abs(self.skew_df[0]) >= self.thresh]
        skewed_columns = skewed_columns["index"].to_list()
        # Return all skewed columns if no manual mappings are
        if self.manual_mapping is None:
            return list(set(skewed_columns))
        # Remove the skewed columns which are getting manual transformation applied to them
        skewed_columns = list(set(skewed_columns) - set(self.manual_mapping.keys()))
        return skewed_columns

    def get_positive_columns(self, skewed_columns):
        """
        Get a list of columns which contain only positive values
        for box cox transform.

        Args:
            skewed_columns (list): list of skewed columns 

        Returns:
            positive_columns (list): A list of columns which contain only positive values
        """
        positive_columns = self.data[skewed_columns].loc[:, self.data[skewed_columns].gt(0).all()]
        positive_columns = list(positive_columns.columns)
        return positive_columns

    def post_transform_skew(self):
        """
        Performs skew transformations on all columns to determine
        the best transform.

        Performs all transformations on all columns and generates
        a DataFrame with details of the skew data.

        Returns:
            df (DataFrame): A DataFrame which has the values of 
        """
        skewed_columns = self.get_skewed_columns()
        positive_columns = self.get_positive_columns(skewed_columns)
        log_df = self.perform_transform(skewed_columns, self.log_transform)
        sqrt_df = self.perform_transform(skewed_columns, self.sqrt_transform)
        box_cox_df = self.perform_transform(positive_columns, self.box_cox_transform)
        cube_transform = self.perform_transform(skewed_columns, self.cube_transform)
        df = pd.concat([log_df, sqrt_df], axis=1)
        df = pd.concat([df, box_cox_df], axis=1)
        df = pd.concat([df, cube_transform], axis=1)
        df.columns = ["log", "sqrt", "box_cox", "cube"]
        df = df.abs()
        df["Min"] = df.idxmin(axis=1)
        return df

    def extract_transform_type(self, transform_type):
        """Extract the transform type with results in
           minimum skew on a column.

        Args:
            transform_type (str): The transform type 
                to extract the rows in the DataFrame.

        Returns:
            columns (list): List of columns where the 
                transform has resulted in the minimum amount of skew.
        """
        df = self.post_transform_skew()
        # Extract the rows in the DataFrame where the minimum is of that type of transform
        columns = df.loc[df["Min"] == transform_type]
        columns = columns.index.to_list()
        return columns
    
    def perform_transform_get_results(self, column, transform_function):
        prev_skew = round(self.data[column].skew(), 2)
        self.data[column] = self.data[column].apply(transform_function)
        skew_change = round(self.data[column].skew(), 2)
        skew_change = [prev_skew, skew_change]
        return skew_change

    def transform_data(self):
        # TODO change this to work on the result of a dictionary mapping to reduce size
        """transform all the columns in the DataFrame to reduce the skew

        Returns:
           skew_results (list): List of results of the automatically transformed columns.
           manual_results (list): List of results of the manually transformed columns, 
        """
        # Check the columns to see which transform results in the lowest skew
        manual_results = self.perform_manual_skew()
        log_columns = self.extract_transform_type("log")
        sqrt_columns = self.extract_transform_type("sqrt")
        box_cox_columns = self.extract_transform_type("box_cox")
        cube_columns = self.extract_transform_type("cube")
        skew_results = []
        # Perform the skew transforms on the columns to result in the minimum skew amount
        # NOTE not using two loops here for performance
        for column in log_columns:
            skew_change = self.perform_transform_get_results(column, self.log_transform)
            skew_results.append(self.build_skew_string(skew_change, column, 'LOG'))
        for column in sqrt_columns:
            skew_change = self.perform_transform_get_results(column, self.sqrt_transform)
            skew_results.append(self.build_skew_string(skew_change, column, 'SQRT'))       
        for column in box_cox_columns:
            prev_skew = round(self.data[column].skew(), 2)
            self.data[column] = self.box_cox_transform(self.data[column])
            skew_change = round(self.data[column].skew(), 2)
            skew_change = [prev_skew, skew_change]
            skew_results.append(self.build_skew_string(skew_change, column, 'BOX_COX'))
        for column in cube_columns:
            skew_change = self.perform_transform_get_results(column, self.cube_transform)
            skew_results.append(self.build_skew_string(skew_change, column, 'CUBED'))
        return skew_results, manual_results

    @staticmethod    
    def build_skew_string(data, column, transform):
        """Build a string representing the skew performed and the column the skew was performed on

        Args:
            data (list): List containing the skew before and after values.
            column (str): Column name the skew was performed on.
            transform (str): The type of skew transform applied to the column
        """
        string = f"Column {column.upper()} skew was changed from {data[0]} to {data[1]} using transform {transform}"
        return string
    
    def sqrt_transform(self, value):
        """Perform squaring of the column values

        Args:
            value (float): Cell value to perform the transformation on.

        Returns:
            (float): Returns float value with the log transformed. 
        """
        if value > 0:
            return np.sqrt(value)
        else:
            return 0

    def log_transform(self, value):
        if value > 0:
            return np.log(value)
        else:
            return 0

    def cube_transform(self, value):
        """Cube values in the DataFrame

        Args:
            value (float): 

        Returns:
            (float): _description_
        """
        return np.cbrt(value)

    def box_cox_transform(self, positive_column):
        result = stats.boxcox(positive_column)[0]
        return result

    def perform_transform(self, columns, function):
        df = DataFrame()
        for column in columns:
            if function == self.box_cox_transform:
                result = self.box_cox_transform(self.data[column])
            else:
                result = self.data[column].map(function)
            df[column] = result
        df_skew = df.skew()
        return df_skew