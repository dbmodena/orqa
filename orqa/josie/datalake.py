import pickle
from abc import ABC, abstractmethod

import pandas as pd

class DataLakeHandler(ABC):
    @abstractmethod
    def __init__(self, datalake_location:str, *args):
        pass

    @abstractmethod
    def get_table_by_id(self, _id:str):
        pass

    @abstractmethod
    def get_table_by_numeric_id(self, numeric_id:int):
        pass

    @abstractmethod
    def scan_tables(self, _from:int=0, _to:int=-1):
        pass

    @abstractmethod
    def count_tables(self):
        pass

    @abstractmethod
    def config(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def clone(self):
        pass


class DataLakeHandlerFactory:
    def create_handler(datalake_location:str, *args) -> DataLakeHandler:
        match datalake_location:
            case _:
                return LocalFileDataLakeHandler(datalake_location, *args)


class LocalFileDataLakeHandler(DataLakeHandler):
    def __init__(self, datalake_location, datalake_name, *args):
        self.datalake_location = datalake_location
        self.datalake_name = datalake_name
        self.mapping_id_path = f"{self.datalake_location}/mapping_id.pickle"
        self.valid_columns_path = f"{self.datalake_location}/valid_columns.pickle"
        
        with open(self.mapping_id_path, 'rb') as fr:
            self.mapping_id = pickle.load(fr)
        with open(self.valid_columns_path, 'rb') as fr:
            self.valid_columns = pickle.load(fr)

    def get_table_by_id(self, _id):
        raise NotImplementedError()

    def get_table_by_numeric_id(self, _id_numeric):
        # content = pl.read_csv(f'{self.datalake_location}/tables/{self.mapping_id[_id_numeric]}.csv', has_header=False, infer_schema_length=0, encoding='latin1').rows()
        content = pd.read_parquet(f'{self.datalake_location}/tables/{self.mapping_id[_id_numeric]}.parquet', encoding='latin-1').values.tolist()
        
        valid_columns = self.valid_columns[_id_numeric]
        headers = content[0]
        return {'_id_numeric': _id_numeric, 'content': content, 'headers': headers, 'valid_columns': valid_columns}

    def count_tables(self):
        return len(self.mapping_id)
        
    def scan_tables(self, _from:int = 0, _to:int = -1):
        for _id_numeric in range(_from, _to + 1):
            yield self.get_table_by_numeric_id(_id_numeric)

    def config(self):
        return self.datalake_location, self.datalake_name

    def close(self):
        pass

    def clone(self):
        return DataLakeHandlerFactory.create_handler(self.datalake_location, self.datalake_name)



