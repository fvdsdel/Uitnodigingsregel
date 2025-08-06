# if __name__ == "__main__":
#     from oracle import OraDB

from .oracle import OraDB
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from dotenv import load_dotenv, dotenv_values

VERSIE = "6"


# loading variables from .env file
DEFAULT_CWDIR = Path(__file__).parent

query_path = "osiris_queries"
file_inschrijvingen = f"./{query_path}/inschrijvingen.sql"
file_verzuim = f"./{query_path}/verzuim.sql"
file_vooropleiding_diploma = f"./{query_path}/vopl_diplomas.sql"
file_dossiers = f"./{query_path}/dossiers.sql"

VZCOLS = [
    "STUDENTNUMMER","SCHOOLJAAR"
    ,"TOTAALAANW3","TOTAALAANW6","TOTAALAANW10"
    # ,"TOTAALREG3","TOTAALREG6","TOTAALREG10"
    ,"AANW_PCT3","AANW_PCT6","AANW_PCT10"]


OUPUT_COLS = ["STUDENTNUMMER"
              ,"STUD_GENDER_M","STUD_GENDER_V","STUD_GENDER_O","LEEFTIJD"
            ,"SOPL_LW_BOL","SOPL_LW_BBL","SOPL_NIV1","SOPL_NIV2","SOPL_NIV3","SOPL_NIV4"
            ,"VOOROPLNIVEAU_NAN","VOOROPLNIVEAU_HAVO","VOOROPLNIVEAU_VMBO_BB","VOOROPLNIVEAU_VMBO_GL","VOOROPLNIVEAU_VMBO_KB"
            ,"VOOROPLNIVEAU_VMBO_TL","VOOROPLNIVEAU_VWO","VOOROPLNIVEAU_MBO","VOOROPLNIVEAU_HO"
            ,"TOTAALAANW3","TOTAALAANW6","TOTAALAANW10","AANW_PCT3","AANW_PCT6","AANW_PCT10"
            ,"GEEN_UITVAL"
            ,"TEAM"
            ]

def normalize_df(df):
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(x_scaled)
    df_normalized.columns = df.columns
    return df_normalized

def fill_empty_cols(df:pd.DataFrame, cols:list) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    return df

class OsirisData:
    def __init__(self,cwd=DEFAULT_CWDIR,env_path=None):
        if not env_path:
            env_path=cwd
        if isinstance(env_path,str):
            env_path = Path(env_path)
        load_dotenv(dotenv_path=env_path / ".env") 
        self._cwd = cwd
        self.get_db_connection()
        self.df_insch = None
        self.df_vopl = None
        self.df_verz = None
        self.df_totaal = None
        print("__ Init done __")
    
    def get_db_connection(self):
        self.db_conn = OraDB(user=os.getenv("user",""),passwd=os.getenv("pass",""),host=os.getenv("host",""),port=int(os.getenv("port","1")),srv_name=os.getenv("srv_name",""))
    
    def get_inschrijvingen_data(self, path=file_inschrijvingen,team=None,min_cohort=2022,max_cohort=2024) -> pd.DataFrame:
        p_where = f"1=1 AND SOPL_COHORT >= {min_cohort} AND SOPL_COHORT <= {max_cohort}"
        if team is not None:
            if len(team)>2:
                p_where = p_where + f"AND team = '{team}'"
            else:
                p_where = p_where + f"AND team LIKE '%{team}'"
        qpath = self._cwd/path
        df = self.db_conn.run_query_from_file(qpath,where=p_where)
        # df_inschrijvingen = df_inschrijvingen.drop_duplicates().reset_index(drop=True)
        return df
    
    def get_verzuim_data(self, path=file_verzuim,p_where=None) -> pd.DataFrame:
        qpath = self._cwd/path
        df = self.db_conn.run_query_from_file(qpath,where=p_where)
        return df
    
    def get_vooropleiding_data(self, path=file_vooropleiding_diploma) -> pd.DataFrame:
        qpath = self._cwd/path
        df = self.db_conn.run_query_from_file(qpath)
        return df
        
    def get_dossier_data(self,path=file_dossiers) -> pd.DataFrame:
        df_dossiers = self.db_conn.run_query_from_file(path)
        df_dossiers = df_dossiers[["OPLEIDING_CREBO","DOSSIER","KWALIFICATIE","DOMEIN"]]
        df_dossiers = df_dossiers.drop_duplicates().reset_index(drop=True)
        return df_dossiers
    
    def clean_vooropleidingdata(self,df:pd.DataFrame) -> pd.DataFrame:
        df["VOOROPLNIVEAU_NAN"].fillna(1,inplace=True)
        vopl_cols = [x for x in df.columns if "VOOROPLNIVEAU" in x and x != "VOOROPLNIVEAU_NAN"]
        df = fill_empty_cols(df, vopl_cols)
        return df
    
    def load_data_from_db(self,team=None,min_cohort=2022,max_cohort=2024) -> pd.DataFrame:
        """  """
        self.df_insch = self.get_inschrijvingen_data(team=team,min_cohort=min_cohort,max_cohort=max_cohort)
        print("inschrijvingen geladen...")
        self.df_verz = self.get_verzuim_data()
        self.df_verz = self.df_verz[VZCOLS]
        print("verzuimdata geladen...")
        self.df_vopl = self.get_vooropleiding_data()
        print("vooropleidingdata geladen...")
        
        df = self.df_insch.merge(self.df_vopl, left_on=["STUDENTNUMMER"],right_on=["STUDENTNUMMER"], how="left")
        df = self.clean_vooropleidingdata(df)

        df = df.merge(self.df_verz, left_on=["STUDENTNUMMER","STARTJAAR"],right_on=["STUDENTNUMMER","SCHOOLJAAR"], how="left")
        # df = df[OUPUT_COLS]
        self.df_totaal = df
        return df

    def get_dataset(self,output_cols:list = OUPUT_COLS,team=None,min_cohort=2022,max_cohort=2024):
        df = self.load_data_from_db(team=team,min_cohort=min_cohort,max_cohort=max_cohort)
        cols = [x for x in output_cols if x in df.columns]
        df = df[cols]
        df["dropout"] = df["GEEN_UITVAL"].apply(lambda x: 0 if x==1 else 1)
        return df

if __name__ == "__main__":
    od = OsirisData(env_path="C:\\Users\\fritssteege\\Documents\\Python\\DGO-WG3\\Uitnodigingsregel")

    # team="BB"

    # df = od.get_dataset(team=team,min_cohort=2022,max_cohort=2024)
    # df.to_csv(f"output_{team}_train.csv")
    # writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')
    # df.to_excel(excel_writer=writer,sheet_name="all_data")
    # od_pred = OsirisDataPipeline(min_cohort=2025,max_cohort=2025)
 
    # df_pred = od.get_dataset(team=team,min_cohort=2025,max_cohort=2025)
    # df_pred.to_csv(f"output_{team}_pred.csv")