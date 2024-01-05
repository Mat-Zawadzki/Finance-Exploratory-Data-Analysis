#%%
import sqlalchemy
import psycopg2
import yaml
import pandas as pd



class DatabaseEngine:

    def __init__(self, cred_file, db_api):
        self.creds = self.load_creds(cred_file)
        self.db_api = db_api
        self.engine = self.init_db_engine()

    def load_creds(self, cred_file):
        '''Load the credentials file for access to the database

        Args:
            cred_file: path of the credentials file to connect to the database

        Returns:
            creds: returns a dict containing the credentials to
            connect to the database.
        '''
        with open(cred_file, 'r') as outfile:
            creds = yaml.safe_load(outfile)
        print(type(creds))
        return creds

    def init_db_engine(self):
        '''Initialise the engine to connect to the database

        Args:
            db_api: database api to use to connect to the database

        returns:
            Returns an sqlalchemy engine to interact with the database
        '''
        creds = self.creds
        HOST = creds["DB_HOST"]
        PORT = creds["DB_PORT"]
        PASS = creds["DB_PASS"]
        USERNAME = creds["DB_USERNAME"]
        DB_NAME = creds["DB_NAME"]

        engine = sqlalchemy.create_engine( f"{self.db_api}+psycopg2://{USERNAME}:{PASS}@{HOST}:{PORT}/{DB_NAME}")
        return engine

'''
    def upload_table(self, dataframe: pd.DataFrame, table_name):
        
        Uses an sqlalchemy engine to upload data to a database

        Args:
            dataframe: passed dataframe containing
                data to upload to the database
            table_name: Name of the table to save the data to.

        
        dataframe.to_sql(table_name, self.engine, if_exists="replace", index=False)
        return
'''
    def table_to_dataframe(self, table_name):
        df = pd.read_sql_table(table_name, self.engine)
        return df


if __name__ == "__main__":
    db_conn = DatabaseEngine("credentials.yaml", "postgresql")
    df = db_conn.table_to_dataframe("loan_payments")
    df.to_csv("loan_payments.csv")
#%%