from importlib import resources as impresources
import json
from typing import List

import numpy as np
import pandas as pd

from recurrent_health_events_prediction.data_extraction.data_types import DiseaseType
from recurrent_health_events_prediction import configs
from recurrent_health_events_prediction.data_extraction.utils import assign_charlson_category

class DataExtractorMIMIC:
    def __init__(self, dataset_config: dict, selected_diseases: List[DiseaseType]):
        """
        Initialize the DataExtractor with the path to the data file.
        
        :param data_path: Path to the CSV file containing the data.
        :param selected_diseasess: The disease type to filter the data.
        """
        self.data_path = dataset_config['path']
        self.cols_to_load_config = dataset_config['columns_to_load']
        self.selected_diseases = selected_diseases

        self.patients_with_disease_ids = None
        
        self.admissions_patients_with_disease_df = None
        self.icu_stays_patients_with_disease_df = None
        self.procedures_patients_with_disease_df = None
        self.prescriptions_patients_with_disease_df = None
        self.patients_metadata_df = None

    def _load_admissions_patients_specific_diseases(self):
        """
        Load the admissions data associated with the patients that have at least one of the diseases listed in `self.selected_diseases`.
        """
        selected_diseases = self.selected_diseases

        admissions_df = pd.read_csv(self.data_path + '/ADMISSIONS.csv')
        diagnoses_icd_df = pd.read_csv(self.data_path + '/DIAGNOSES_ICD.csv')
        diagnoses_codes_df = pd.read_csv(self.data_path + '/D_ICD_DIAGNOSES.csv')

        disease_recurrence_df = pd.merge(diagnoses_icd_df, admissions_df[self.cols_to_load_config["admission"]], on='HADM_ID', how='inner')
        disease_recurrence_df = pd.merge(disease_recurrence_df, diagnoses_codes_df[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']], on='ICD9_CODE', how='inner')

        disease_recurrence_df = assign_charlson_category(disease_recurrence_df, icd_column='ICD9_CODE')

        selected_diseases_str_list = [disease.value for disease in selected_diseases]
        
        subjects_id_s = disease_recurrence_df[disease_recurrence_df['COMORBIDITY'].isin(selected_diseases_str_list)]["SUBJECT_ID"].unique()
        self.patients_with_disease_ids = subjects_id_s

        disease_recurrence_df = disease_recurrence_df[disease_recurrence_df['SUBJECT_ID'].isin(subjects_id_s)]

        disease_recurrence_df['ADMITTIME'] = pd.to_datetime(disease_recurrence_df['ADMITTIME'])
        disease_recurrence_df['DISCHTIME'] = pd.to_datetime(disease_recurrence_df['DISCHTIME'])
        
        self.admissions_patients_with_disease_df = disease_recurrence_df

    def _load_icu_stays_patients_specific_diseases(self):
        """
        Load the ICU stays associated with the patients that have at least one of the diseases listed in `self.selected_diseases`.
        """
        icu_stays_df = pd.read_csv(self.data_path + '/ICUSTAYS.csv')
        icu_stays_df['INTIME'] = pd.to_datetime(icu_stays_df['INTIME'])
        icu_stays_df['OUTTIME'] = pd.to_datetime(icu_stays_df['OUTTIME'])

        subjects_id_s = self.patients_with_disease_ids
        icu_stays_df = icu_stays_df[icu_stays_df['SUBJECT_ID'].isin(subjects_id_s)]

        self.icu_stays_patients_with_disease_df = icu_stays_df[self.cols_to_load_config['icu']]

    def _load_procedures_patients_specific_diseases(self):
        """
        Load the procedures associated with the patients that have at least one of the diseases listed in `self.selected_diseases`.
        """
        procedures_df = pd.read_csv(self.data_path + '/PROCEDURES_ICD.csv')
        procedures_codes_df = pd.read_csv(self.data_path + '/D_ICD_PROCEDURES.csv')

        subjects_id_s = self.patients_with_disease_ids
        procedures_df = procedures_df[procedures_df['SUBJECT_ID'].isin(subjects_id_s)]
        procedures_df = pd.merge(procedures_df, procedures_codes_df[self.cols_to_load_config['procedure']], on='ICD9_CODE', how='left')

        self.procedures_patients_with_disease_df = procedures_df
    
    def _load_prescriptions_patients_specific_diseases(self):
        """
        Load the prescriptions associated with the patients that have at least one of the diseases listed in `self.selected_diseases`.
        """
        prescriptions_df = pd.read_csv(self.data_path + '/PRESCRIPTIONS.csv')

        subjects_id_s = self.patients_with_disease_ids
        prescriptions_df = prescriptions_df[prescriptions_df['SUBJECT_ID'].isin(subjects_id_s)]

        self.prescriptions_patients_with_disease_df = prescriptions_df[self.cols_to_load_config["prescription"]]
    
    def _load_metadata_patients_specific_diseases(self):
        """
        Load the metadata associated with the patients that have at least one of the diseases listed in `self.selected_diseases`.
        """
        patients_df = pd.read_csv(self.data_path + '/PATIENTS.csv')
        patients_df['DOB'] = pd.to_datetime(patients_df['DOB'])
        patients_df['DOD'] = pd.to_datetime(patients_df['DOD'])
        patients_df['DOD_HOSP'] = pd.to_datetime(patients_df['DOD_HOSP'])

        subjects_id_s = self.patients_with_disease_ids
        patients_df = patients_df[patients_df['SUBJECT_ID'].isin(subjects_id_s)]

        self.patients_metadata_df = patients_df[self.cols_to_load_config["patient"]]

    def load_data(self):
        """
        Load the data from the CSV files.
        """
        self._load_admissions_patients_specific_diseases()
        self._load_icu_stays_patients_specific_diseases()
        self._load_procedures_patients_specific_diseases()
        self._load_prescriptions_patients_specific_diseases()
        self._load_metadata_patients_specific_diseases()
    
    def get_admissions_df(self):
        """
        Get all  of patients that were diagnosticated at least once with the specific disease. 
        
        :return: DataFrame containing the admissions data for the specific disease.
        """
        if self.admissions_patients_with_disease_df is None:
            self.load_data()
        return self.admissions_patients_with_disease_df
    
    def get_icu_stays_df(self):
        """
        Get the ICU stays data for the admissions.
        
        :return: DataFrame containing the ICU stays data for the admissions.
        """
        if self.icu_stays_patients_with_disease_df is None:
            self.load_data()
        return self.icu_stays_patients_with_disease_df
    
    def get_procedures_df(self):
        """
        Get the procedures data for the admissions.
        
        :return: DataFrame containing the procedures data for the admissions.
        """
        if self.procedures_patients_with_disease_df is None:
            self.load_data()
        return self.procedures_patients_with_disease_df
    
    def get_prescriptions_df(self):
        """
        Get the prescriptions data for the admissions.
        
        :return: DataFrame containing the prescriptions data for the admissions.
        """
        if self.prescriptions_patients_with_disease_df is None:
            self.load_data()
        return self.prescriptions_patients_with_disease_df
    
    def get_patients_df(self):
        """
        Get the patients data.
        
        :return: DataFrame containing the patients data.
        """
        if self.patients_metadata_df is None:
            self.load_data()
        return self.patients_metadata_df

class DataExtractorDrugRelapse:
    def __init__(self, dataset_config: dict):
        """
        Initialize the DataExtractor for drug relapse data.
        
        :param data_path: Path to the CSV file containing the drug relapse data.
        """
        self.data_path = dataset_config['path']
        self.drug_tests_df = None
        self.donors_df = None
        self.drop_no_showedup = dataset_config.get('drop_no_showedup', True)
        self.min_num_test_days = dataset_config.get('min_num_test_days', 3)

    def load_data(self):
        """
        Load the drug tests data from the CSV file.
        """
        drug_tests_df = pd.read_csv(self.data_path + '/events_averhealth.csv', low_memory=True)
        drug_tests_df['time'] = pd.to_datetime(drug_tests_df['time'])
        drug_tests_df['showedup'] = drug_tests_df['showedup'].astype(bool)

        donors_df = pd.read_csv(self.data_path + '/donors_averhealth.csv', low_memory=True)

        drug_tests_df['donor_id'] = pd.to_numeric(drug_tests_df['donor_id'], errors='coerce')
        drug_tests_df = drug_tests_df.dropna(subset=['donor_id'])
        drug_tests_df['donor_id'] = drug_tests_df['donor_id'].astype(int)

        donors_df['donor_id'] = pd.to_numeric(donors_df['donor_id'], errors='coerce')
        donors_df = donors_df.dropna(subset=['donor_id'])
        donors_df['donor_id'] = donors_df['donor_id'].astype(int)

        drug_tests_df = self._filter_no_showedup(drug_tests_df)
        drug_tests_df, donors_df = self._filter_patients(drug_tests_df, donors_df)

        self.drug_tests_df = drug_tests_df
        self.donors_df = donors_df
    
    def _filter_patients(self, drug_tests_df: pd.DataFrame, donors_df: pd.DataFrame):
        """
        Filter patients based on the number of drug tests and the number of positive drug tests.
        """
        min_num_test_days = self.min_num_test_days
        print("Removing patients always positive or negative and with test days below minimum...")
        print(f"Minimum number of test days: {min_num_test_days}")
        # Get the overall statistics for each donor
        drug_tests_grouped_df = drug_tests_df.groupby(by=["donor_id", "time"]).agg({
            'drug_test_positive': 'any',
        }).reset_index()

        overall_stats_df = drug_tests_grouped_df.groupby('donor_id').agg({
            'drug_test_positive': ['mean', 'count']
        }).reset_index()
        overall_stats_df.columns = ['donor_id', 'positive_rate', 'num_test_days']

        filter_mask = (overall_stats_df['positive_rate'] > 0.0) & (overall_stats_df['positive_rate'] < 1.0) & (overall_stats_df['num_test_days'] >= min_num_test_days)

        subject_ids = overall_stats_df[filter_mask]['donor_id'].unique()

        drug_tests_df = drug_tests_df[drug_tests_df['donor_id'].isin(subject_ids)]
        donors_df = donors_df[donors_df['donor_id'].isin(subject_ids)]
        
        return drug_tests_df, donors_df
    
    def _filter_no_showedup(self, drug_tests_df: pd.DataFrame):
        """
        Filter out the drug tests where the patient did not show up.
        
        :param drug_tests_df: DataFrame containing the drug tests data.
        :return: Filtered DataFrame with only the showed up drug tests.
        """
        print("Removing drug tests where the patient did not show up...")
        if self.drop_no_showedup:
            return drug_tests_df[drug_tests_df['showedup']]
        return drug_tests_df

    def get_drug_tests_df(self):
        """
        Get the drug tests data.
        
        :return: DataFrame containing the drug tests data.
        """
        if self.drug_tests_df is None:
            self.load_data()
        return self.drug_tests_df
    
    def get_donor_df(self):
        """
        Get the donor data from the drug tests.
        
        :return: DataFrame containing the donor data.
        """
        if self.donors_df is None:
            self.load_data()
        
        return self.donors_df
    
