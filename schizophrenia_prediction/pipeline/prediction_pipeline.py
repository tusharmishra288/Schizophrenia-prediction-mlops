import os
import sys

import numpy as np
import pandas as pd
from schizophrenia_prediction.entity.config_entity import SchizophreniaPredConfig
from schizophrenia_prediction.entity.s3_estimator import SchizophreniaEstimator
from schizophrenia_prediction.exception import SchizophreniaPredException
from schizophrenia_prediction.logger import logging
from schizophrenia_prediction.utils.main_utils import read_yaml_file
from pandas import DataFrame


class SchizophreniaData:
    def __init__(self,
                Disease_Duration,
                Hospitalizations,
                Family_History,
                Substance_Use,
                Suicide_Attempt,
                Positive_Symptom_Score,
                Negative_Symptom_Score,
                GAF_Score,
                Medication_Adherence
                ):
        """
        schizophrenia Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Disease_Duration = Disease_Duration
            self.Hospitalizations = Hospitalizations
            self.Family_History = Family_History
            self.Substance_Use = Substance_Use
            self.Suicide_Attempt = Suicide_Attempt
            self.Positive_Symptom_Score = Positive_Symptom_Score
            self.Negative_Symptom_Score = Negative_Symptom_Score
            self.GAF_Score = GAF_Score
            self.Medication_Adherence = Medication_Adherence


        except Exception as e:
            raise SchizophreniaPredException(e, sys) from e

    def get_schizophrenia_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from schizophreniaData class input
        """
        try:
            
            schizophrenia_input_dict = self.get_schizophrenia_data_as_dict()
            return DataFrame(schizophrenia_input_dict)
        
        except Exception as e:
            raise SchizophreniaPredException(e, sys) from e


    def get_schizophrenia_data_as_dict(self):
        """
        This function returns a dictionary from SchizophreniaData class input 
        """
        logging.info("Entered get_schizophrenia_data_as_dict method as SchizophreniaData class")

        try:
            input_data = {
                "Disease_Duration": [self.Disease_Duration],
                "Hospitalizations": [self.Hospitalizations],
                "Family_History": [self.Family_History],
                "Substance_Use": [self.Substance_Use],
                "Suicide_Attempt": [self.Suicide_Attempt],
                "Positive_Symptom_Score": [self.Positive_Symptom_Score],
                "Negative_Symptom_Score": [self.Negative_Symptom_Score],
                "GAF_Score": [self.GAF_Score],
                "Medication_Adherence": [self.Medication_Adherence]
            }  

            logging.info(input_data)

            logging.info("Created schizophrenia data dict")

            logging.info("Exited get_schizophrenia_data_as_dict method as schizophreniaData class")

            return input_data

        except Exception as e:
            raise SchizophreniaPredException(e, sys) from e

class SchizophreniaClassifier:
    def __init__(self,prediction_pipeline_config: SchizophreniaPredConfig = SchizophreniaPredConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise SchizophreniaPredException(e, sys)


    def predict(self, dataframe) -> str:
        """
        This is the method of schizophreniaClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of schizophreniaClassifier class")
            model = SchizophreniaEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise SchizophreniaPredException(e, sys)