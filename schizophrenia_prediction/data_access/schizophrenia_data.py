from schizophrenia_prediction.configuration.mongo_db_connection import MongoDBClient
from schizophrenia_prediction.constants import DATABASE_NAME,FILE_NAME,FILE_SOURCE_PATH
from schizophrenia_prediction.exception import SchizophreniaPredException
import pandas as pd
import sys
from typing import Optional
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter



class SchizophreniaData:
    """
    This class help to export entire mongo db record as pandas dataframe
    """

    def __init__(self):
        """
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise SchizophreniaPredException(e,sys)
        

    def export_collection_as_dataframe(self,collection_name:str,database_name:Optional[str]=None)->pd.DataFrame:
        try:
            """
            export entire collectin as dataframe:
            return pd.DataFrame of collection
            """
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

            # Load the latest version from kaggle 
            df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            FILE_SOURCE_PATH,
            FILE_NAME
            )

            #renaming the columns 
            df.rename(columns={"Hasta_ID":"Patient_ID",
                        "Yaş":"Age",
                        "Cinsiyet":"Gender",
                        "Eğitim_Seviyesi":"Education_level",
                       "Medeni_Durum":"Marital_Status",
                       "Meslek":"Occupation",
                       "Gelir_Düzeyi":"Income_level",
                       "Yaşadığı_Yer":"Living_Area",
                       "Tanı":"Diagnosis",
                       "Hastalık_Süresi":"Disease_Duration",
                       "Hastaneye_Yatış_Sayısı":"Hospitalizations",
                       "Ailede_Şizofreni_Öyküsü":"Family_History",
                       "Madde_Kullanımı":"Substance_Use",
                       "İntihar_Girişimi":"Suicide_Attempt",
                       "Pozitif_Semptom_Skoru":"Positive_Symptom_Score",
                       "Negatif_Semptom_Skoru":"Negative_Symptom_Score",
                       "GAF_Skoru":"GAF_Score",
                       "Sosyal_Destek":"Social_Support",
                       "Stres_Faktörleri":"Stress_Factors",
                       "İlaç_Uyumu":"Medication_Adherence"}
                       ,inplace = True)
            
            data = df.to_dict(orient='records')

            cursor =  collection.find()
            record_length = []
            for records in cursor:
                record_length.append(records)
                
            if len(record_length) == 0:
                collection.insert_many(data)
            else:
                collection.delete_many({})
                collection.insert_many(data)    

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise SchizophreniaPredException(e,sys)