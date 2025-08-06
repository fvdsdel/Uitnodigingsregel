import oracledb
import pandas as pd
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
_cwd = Path(__file__).parent
oic_path = str(_cwd/"instantclient_23_8")
oracledb.init_oracle_client(lib_dir=oic_path)


class OraDB:
    def __init__(self,user:str='user',passwd:str='passwd',host:str='127.0.0.1',port:int=1522,srv_name:str='PRDS'):
        self._host = host
        self._port = port
        self._name = srv_name
        self._user = user
        self._pass = passwd
        self._dsn = self.get_dsn()
    
    def get_dsn(self):
        """Returns a DSN string"""	
        return oracledb.makedsn(self._host,self._port,service_name=self._name)
    
    def get_connection(self):
        """Returns a connection object"""	
        return oracledb.connect(user=self._user, password=self._pass, dsn=self._dsn)
    
    def get_cursor(self):
        """ Returns a cursor object for the connection"""
        self.get_connection.cursor()
    
    def run_query(self,q,index_col=None):
        """Runs a query and returns a pandas dataframe"""
        conn = self.get_connection()
        df = pd.read_sql(q, conn,index_col=None)
        conn.close()
        if index_col:
            df.set_index(index_col,inplace=True)
        return df
    
    def run_query_from_file(self,file,index_col=None,where=None):
        """Reads a query from a file and runs it"""	
        with open(file,'r') as f:
            q = f.read()
        if where:
            q = q + "\n AND " + where
        return self.run_query(q,index_col=index_col)

