import pandas as pd
import numpy as np
import json
from importlib import resources as impresources
from recurrent_health_events_prediction import configs
from recurrent_health_events_prediction.data_extraction.data_types import DiseaseType
from recurrent_health_events_prediction.preprocessing.utils import bin_time_col_into_cat, calculate_past_rolling_stats, calculate_past_rolling_stats_multiple_features

with open((impresources.files(configs) / 'comorbidity_charlson_weights.json')) as f:
        charlson_weights = json.load(f)

class FeatureExtractorMIMIC:
    @classmethod
    def build_features(cls, admissions_df: pd.DataFrame, icu_stays_df: pd.DataFrame,
                       prescriptions_df: pd.DataFrame, procedures_df: pd.DataFrame,
                       patients_metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build features for the given disease recurrence DataFrame.
        """
        
        # Calculate time in ICU for each patient for every admission
        icu_stay_features_df = cls._calculate_days_in_icu(icu_stays_df)
        
        # Calculate number of drugs prescribed for each patient for every admission
        prescriptions_features_df = cls._calculate_num_drugs_prescribed(prescriptions_df)
        
        # Calculate number of procedures performed for each patient for every admission
        procedures_features_df = cls._calculate_num_procedures(procedures_df)

        # Calculate features related to the admission
        admissions_features_df = cls._get_admission_related_features(admissions_df)

        # Order the admissions of every patient in time
        admissions_features_df.sort_values(by=['SUBJECT_ID', 'ADMITTIME'], inplace=True)

        # Get the next admission type for each patient for every admission
        admissions_features_df = cls._get_next_admission_type(admissions_features_df)
        
        admissions_features_df = cls._get_time_hospitalization_stats(admissions_features_df)

        # Filter out events with negative days until next hospitalization or days since last hospitalization
        invalid_mask = (admissions_features_df['DAYS_UNTIL_NEXT_HOSPITALIZATION'] < 0) | (admissions_features_df['DAYS_SINCE_LAST_HOSPITALIZATION'] < 0)
        admissions_features_df = admissions_features_df[~invalid_mask]

        admissions_features_df = cls._calculate_past_hospitalization_stats(admissions_features_df)

        # Get the total number of hospitalizations for each patient
        admissions_features_df = cls._get_total_number_of_hospitalizations(admissions_features_df)
        
        # Merge all features into the disease recurrence DataFrame
        merged_df = pd.merge(admissions_features_df, icu_stay_features_df, on=['SUBJECT_ID', 'HADM_ID'], how='left')
        merged_df = pd.merge(merged_df, prescriptions_features_df, on=['SUBJECT_ID', 'HADM_ID'], how='left')
        merged_df = pd.merge(merged_df, procedures_features_df, on=['SUBJECT_ID', 'HADM_ID'], how='left')

        # Add patient specific features such as age
        merged_df = cls._add_patient_specific_features(merged_df, patients_metadata_df)

        # Calculate Charlson Comorbidity Index
        merged_df = cls._get_patient_comorbidity_index(merged_df)

        # Get the number of days since the first admission and each of its discharges
        merged_df = cls._get_known_participation_days(merged_df)
        # Get the total participation days until patient discharge for every admission
        merged_df = cls._get_known_total_participation_days(merged_df)

        merged_df = cls._get_death_time_after_last_discharge(merged_df)

        # Fill NaN values with 0 and convert types
        merged_df['DAYS_IN_ICU'] = merged_df['DAYS_IN_ICU'].fillna(0.0).astype(float)
        merged_df['NUM_DRUGS'] = merged_df['NUM_DRUGS'].fillna(0).astype(int)
        merged_df['NUM_PROCEDURES'] = merged_df['NUM_PROCEDURES'].fillna(0).astype(int)
        merged_df['NUM_PREV_HOSPITALIZATIONS'] = merged_df['NUM_PREV_HOSPITALIZATIONS'].fillna(0).astype(int)
        merged_df['NUM_COMORBIDITIES'] = merged_df['NUM_COMORBIDITIES'].fillna(0).astype(int)
        merged_df['CHARLSON_INDEX'] = merged_df['CHARLSON_INDEX'].fillna(0).astype(int)
        merged_df['PARTICIPATION_DAYS'] = merged_df['PARTICIPATION_DAYS'].astype(int)
        merged_df['TOTAL_PARTICIPATION_DAYS'] = merged_df['TOTAL_PARTICIPATION_DAYS'].astype(int)
        merged_df['PREV_READMISSION_30_DAYS'] = merged_df['PREV_READMISSION_30_DAYS'].fillna(-1).astype(int)
        merged_df['READMISSION_30_DAYS'] = merged_df['READMISSION_30_DAYS'].fillna(0).astype(int)

        return merged_df

    @classmethod
    def _calculate_days_in_icu(cls, icu_stay_df: pd.DataFrame):
        """
        Calculate the time spent in ICU for each patient for every admission.
        """
                                  
        icu_stay_df['DAYS_IN_ICU'] = (icu_stay_df['OUTTIME'] - icu_stay_df['INTIME']).dt.total_seconds() / 3600 / 24 # Convert to days
        icu_stay_df['DAYS_IN_ICU'] = icu_stay_df['DAYS_IN_ICU'].fillna(0.0)

        icu_stay_df = icu_stay_df.groupby(['SUBJECT_ID', 'HADM_ID']).agg({'DAYS_IN_ICU': 'sum'}).reset_index()
        return icu_stay_df
    
    @classmethod
    def _calculate_num_drugs_prescribed(cls, prescriptions_df: pd.DataFrame):
        """
        Calculate the number of distinct drugs prescribed for each patient for every admission.
        """        
        prescriptions_df = prescriptions_df.groupby(['SUBJECT_ID', 'HADM_ID']).agg(NUM_DRUGS=("DRUG", "nunique")).reset_index()
        prescriptions_df['NUM_DRUGS'] = prescriptions_df['NUM_DRUGS'].fillna(0)
        prescriptions_df['NUM_DRUGS'] = prescriptions_df['NUM_DRUGS'].astype(int)
        return prescriptions_df
    
    @classmethod
    def _calculate_num_procedures(cls, procedures_df: pd.DataFrame):
        """
        Calculate the number of distinct procedures performed for each patient for every admission.
        """        
        procedures_df = procedures_df.groupby(['SUBJECT_ID', 'HADM_ID']).agg(NUM_PROCEDURES = ("ICD9_CODE", "nunique")).reset_index()
        procedures_df['NUM_PROCEDURES'] = procedures_df['NUM_PROCEDURES'].fillna(0)
        procedures_df['NUM_PROCEDURES'] = procedures_df['NUM_PROCEDURES'].astype(int)
        return procedures_df
    
    @classmethod
    def _get_time_hospitalization_stats(cls, admissions_df: pd.DataFrame):
        """
        Calculate the time statistics related to hospitalizations for each patient.
        This includes:
        - Number of previous hospitalizations
        - Days since last hospitalization (log scale as well)
        - Days until next hospitalization (log scale as well)
        - Whether there was a readmission within 30 days before and after the current hospitalization
        - PREV_DISCHTIME: the discharge time of the previous hospitalization
        - NEXT_ADMITTIME: the admission time of the next hospitalization
        This function assumes that the admissions_df DataFrame is sorted by ADMITTIME.
        """
        # Assign sequence number for hospital visits
        admissions_df['NUM_PREV_HOSPITALIZATIONS'] = admissions_df.groupby('SUBJECT_ID').cumcount()

        # Calculate days since last hospitalization
        admissions_df['PREV_DISCHTIME'] = admissions_df.groupby('SUBJECT_ID')['DISCHTIME'].shift(1)
        admissions_df['NEXT_ADMITTIME'] = admissions_df.groupby('SUBJECT_ID')['ADMITTIME'].shift(-1)
        admissions_df['DAYS_SINCE_LAST_HOSPITALIZATION'] = (admissions_df['ADMITTIME'] - admissions_df['PREV_DISCHTIME']).dt.total_seconds() / 3600 / 24
        admissions_df['DAYS_UNTIL_NEXT_HOSPITALIZATION'] = (admissions_df['NEXT_ADMITTIME'] - admissions_df['DISCHTIME']).dt.total_seconds() / 3600 / 24

        # Clipping negative values to zero
        admissions_df['DAYS_SINCE_LAST_HOSPITALIZATION'] = admissions_df['DAYS_SINCE_LAST_HOSPITALIZATION'].clip(lower=0)
        admissions_df['DAYS_UNTIL_NEXT_HOSPITALIZATION'] = admissions_df['DAYS_UNTIL_NEXT_HOSPITALIZATION'].clip(lower=0)

        admissions_df['LOG_DAYS_SINCE_LAST_HOSPITALIZATION'] = np.log1p(admissions_df['DAYS_SINCE_LAST_HOSPITALIZATION'])
        admissions_df['LOG_DAYS_UNTIL_NEXT_HOSPITALIZATION'] = np.log1p(admissions_df['DAYS_UNTIL_NEXT_HOSPITALIZATION'])

        admissions_df['PREV_READMISSION_30_DAYS'] = (admissions_df['DAYS_SINCE_LAST_HOSPITALIZATION'] < 30)
        admissions_df['READMISSION_30_DAYS'] = (admissions_df['DAYS_UNTIL_NEXT_HOSPITALIZATION'] < 30)
        
        admissions_df['PREV_READMISSION_30_DAYS'] = np.where(admissions_df['DAYS_SINCE_LAST_HOSPITALIZATION'].notna(), admissions_df['PREV_READMISSION_30_DAYS'], None)
        admissions_df['READMISSION_30_DAYS'] = np.where(admissions_df['DAYS_UNTIL_NEXT_HOSPITALIZATION'].notna(), admissions_df['READMISSION_30_DAYS'], None)
    
        return admissions_df
    
    @classmethod
    def _calculate_past_hospitalization_stats(cls, admissions_df: pd.DataFrame):
        """
        For every hospitalization 'i', calculate rolling statistics for READMISSION_30_DAYS and DAYS_UNTIL_NEXT_HOSPITALIZATION
        over all past hospitalizations (k = 1 to i-1) of each subject.
        Returns a DataFrame with these statistics, aligned to admissions_df index.
        """
        past_readmission_stats_df = calculate_past_rolling_stats(admissions_df, group_col='SUBJECT_ID', feature='READMISSION_30_DAYS',
                                                                 id_col='HADM_ID', stats=['mean', 'sum'], prefix='READM_30_DAYS')
        past_days_until_next_hosp_stats_df = calculate_past_rolling_stats(admissions_df, group_col='SUBJECT_ID', feature='LOG_DAYS_UNTIL_NEXT_HOSPITALIZATION',
                                                                          id_col='HADM_ID', stats=['mean', 'median', 'std'], prefix='LOG_DAYS_UNTIL_NEXT_HOSP')
        # Merge the past statistics with the original admissions DataFrame
        admissions_df = pd.merge(admissions_df.drop(columns=["SUBJECT_ID"]), past_readmission_stats_df, on='HADM_ID', how='left')
        admissions_df = pd.merge(admissions_df.drop(columns=["SUBJECT_ID"]), past_days_until_next_hosp_stats_df, on='HADM_ID', how='left')
        return admissions_df

    @classmethod
    def _get_admission_related_features(cls, admissions_df):
        """
        Get the type of admission for each patient for every admission.
        """        
        admissions_df['ADMISSION_TYPE'] = admissions_df['ADMISSION_TYPE'].fillna('Unknown')

        admissions_df['ADMITTIME'] = pd.to_datetime(admissions_df['ADMITTIME'])
        admissions_df['DISCHTIME'] = pd.to_datetime(admissions_df['DISCHTIME'])

        admissions_df['HOSPITALIZATION_DAYS'] = (admissions_df['DISCHTIME'] - admissions_df['ADMITTIME']).dt.total_seconds() / 3600 / 24

        admissions_df = admissions_df.groupby(['SUBJECT_ID', 'HADM_ID']).agg(
            ADMITTIME=('ADMITTIME', 'first'),
            DISCHTIME=('DISCHTIME', 'first'),
            ADMISSION_TYPE=('ADMISSION_TYPE', 'first'),
            ETHNICITY=('ETHNICITY', 'first'),
            DISCHARGE_LOCATION=('DISCHARGE_LOCATION', 'first'),
            INSURANCE=('INSURANCE', 'first'),
            HOSPITALIZATION_DAYS=('HOSPITALIZATION_DAYS', 'first'),
            NUM_COMORBIDITIES=('COMORBIDITY', 'nunique'),
            TYPES_COMORBIDITIES=('COMORBIDITY', lambda x: list(set(x))),
            HAS_DIABETES=('COMORBIDITY', lambda x: any('diabetes' in str(c).lower() for c in x)),
            HAS_COPD=('COMORBIDITY', lambda x: any('chronic_pulmonary_disease' in str(c).lower() for c in x)),
            HAS_CONGESTIVE_HF=('COMORBIDITY', lambda x: any('heart_failure' in str(c).lower() for c in x)),

        ).reset_index()
        
        admissions_df['NUM_COMORBIDITIES'] = admissions_df['NUM_COMORBIDITIES'].fillna(0)
        admissions_df['NUM_COMORBIDITIES'] = admissions_df['NUM_COMORBIDITIES'].astype(int)

        return admissions_df

    @classmethod
    def _add_patient_specific_features(cls, hospitalizations_df, patients_metadata_df: pd.DataFrame):
        """
        Get patient specific features such as age
        """
        hospitalizations_df = pd.merge(hospitalizations_df, patients_metadata_df, on='SUBJECT_ID', how='left')
        hospitalizations_df["AGE"] = (hospitalizations_df['ADMITTIME'].dt.year - hospitalizations_df['DOB'].dt.year)

        anonymized_age_mask = (hospitalizations_df['AGE'] >= 90)
        hospitalizations_df.loc[anonymized_age_mask, 'AGE'] = hospitalizations_df.loc[anonymized_age_mask, 'AGE'] - 270

        return hospitalizations_df

    @classmethod
    def _get_total_number_of_hospitalizations(cls, admissions_df: pd.DataFrame):
        """
        Get the total number of hospitalizations for each patient.
        """
        admissions_df['TOTAL_HOSPITALIZATIONS'] = admissions_df.groupby('SUBJECT_ID')['HADM_ID'].transform('nunique')
        return admissions_df

    @classmethod
    def _get_next_admission_type(cls, admissions_df: pd.DataFrame):
        """
        Get the next admission type for each patient for every admission.
        """
        admissions_df['NEXT_ADMISSION_TYPE'] = admissions_df.groupby('SUBJECT_ID')['ADMISSION_TYPE'].shift(-1)
        
        return admissions_df

    @classmethod
    def _get_known_participation_days(cls, admissions_df: pd.DataFrame):
        """
        Get number of days since patient's first admission and each of its discharges 
        """

        admissions_df["FIRST_ADMITTIME"] = admissions_df.groupby("SUBJECT_ID")["ADMITTIME"].transform("min")
        admissions_df["LAST_DISCHTIME"] = admissions_df.groupby("SUBJECT_ID")["DISCHTIME"].transform("max")

        admissions_df["PARTICIPATION_DAYS"] = (admissions_df["DISCHTIME"] - admissions_df["FIRST_ADMITTIME"]).dt.days
        
        admissions_df["PARTICIPATION_DAYS"] = admissions_df["PARTICIPATION_DAYS"].clip(1.0)

        return admissions_df

    @classmethod
    def _get_known_total_participation_days(cls, admissions_df: pd.DataFrame):
        """
        Get the participation days until patient discharge for every admission.
        """
        admissions_df["TOTAL_PARTICIPATION_DAYS"] = np.where(
            admissions_df["DOD"].notna(),
            (admissions_df["DOD"] - admissions_df["FIRST_ADMITTIME"]).dt.days,
            (admissions_df["LAST_DISCHTIME"] - admissions_df["FIRST_ADMITTIME"]).dt.days)
        
        admissions_df["TOTAL_PARTICIPATION_DAYS"] = admissions_df["TOTAL_PARTICIPATION_DAYS"].clip(1.0)

        return admissions_df

    @classmethod
    def _get_death_time_after_last_discharge(cls, admissions_df: pd.DataFrame):
        """
        Get the time of death after the last discharge for each patient.
        """
        admissions_df['DOD'] = pd.to_datetime(admissions_df['DOD'])
        admissions_df['LAST_DISCHTIME'] = pd.to_datetime(admissions_df['LAST_DISCHTIME'])

        admissions_df['DEATH_TIME_AFTER_LAST_DISCHARGE'] = (admissions_df['DOD'] - admissions_df['LAST_DISCHTIME']).dt.total_seconds() / 3600 / 24
        admissions_df['DEATH_TIME_AFTER_LAST_DISCHARGE'] = admissions_df['DEATH_TIME_AFTER_LAST_DISCHARGE'].clip(lower=0)
        admissions_df['DEATH_TIME_AFTER_LAST_DISCHARGE'] = np.where(admissions_df['DEATH_TIME_AFTER_LAST_DISCHARGE'] < 1.0, 0.0, admissions_df['DEATH_TIME_AFTER_LAST_DISCHARGE'])

        return admissions_df

    @classmethod
    def _get_patient_comorbidity_index(cls, admissions_df: pd.DataFrame):
        """
        Given a DataFrame with AGE and TYPES_COMORBIDITIES (a list of comorbidity strings),
        add a CHARLSON_INDEX column .
        Code adaptaded from https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/comorbidity/charlson.sql
        Required columns:
            - AGE
            - TYPES_COMORBIDITIES: list of comorbidity strings (e.g., ['aids', 'renal_disease'])

        Adds:
            - CHARLSON_INDEX: final index score
        """
    
        def age_score(age):
            if age <= 50: return 0
            elif age <= 60: return 1
            elif age <= 70: return 2
            elif age <= 80: return 3
            else: return 4

        def compute_index(row):
            diseases = row["TYPES_COMORBIDITIES"]
            score = age_score(row["AGE"])

            special_diseases = {
                DiseaseType.DIABETES_WITHOUT_COMPLICATION.value,
                DiseaseType.DIABETES_WITH_COMPLICATION.value,
                DiseaseType.MALIGNANT_CANCER.value,
                DiseaseType.METASTATIC_SOLID_TUMOR.value,
                DiseaseType.MILD_LIVER_DISEASE.value,
                DiseaseType.SEVERE_LIVER_DISEASE.value,
            }

            for disease in diseases:
                if disease in charlson_weights and not disease in special_diseases:
                    score += charlson_weights[disease]                

            # GREATEST(mild_liver, 3*severe_liver)
            liver = max(
                int(DiseaseType.MILD_LIVER_DISEASE.value in diseases),
                3 * int(DiseaseType.SEVERE_LIVER_DISEASE.value in diseases)
            )

            # GREATEST(diabetes_without, 2*diabetes_with)
            diabetes = max(
                int(DiseaseType.DIABETES_WITHOUT_COMPLICATION.value in diseases),
                2 * int(DiseaseType.DIABETES_WITH_COMPLICATION.value in diseases)
            )

            # GREATEST(2*malignant, 6*metastatic)
            cancer = max(
                2 * int(DiseaseType.MALIGNANT_CANCER.value in diseases),
                6 * int(DiseaseType.METASTATIC_SOLID_TUMOR.value in diseases)
            )

            return score + liver + diabetes + cancer

        admissions_df["CHARLSON_INDEX"] = admissions_df.apply(compute_index, axis=1)
        return admissions_df

class FeatureExtractorDrugRelapse:
    """
    Feature extractor for drug relapse prediction.
    """

    @classmethod
    def build_features(cls, drug_tests_df: pd.DataFrame, donor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build features for the drug relapse prediction.
        """
        drug_tests_df = cls.uppercase_col_names(drug_tests_df)
        print("Building features for drug relapse prediction...")
        print("Calculating statistics per test day...")
        drug_tests_df = cls._calculate_statistics_per_test_day(drug_tests_df)

        # Ordering the drug tests by DONOR_ID and TIME
        print("Ordering drug tests by DONOR_ID and TIME...")
        drug_tests_df['TIME'] = pd.to_datetime(drug_tests_df['TIME'])
        drug_tests_df = drug_tests_df.sort_values(['DONOR_ID', 'TIME'])

        print("Calculating participation days...")
        drug_tests_df = cls._get_participation_days(drug_tests_df)
        drug_tests_df['PARTICIPATION_DAYS'] = drug_tests_df['PARTICIPATION_DAYS'].clip(1.0)

        print("Calculating time between tests...")
        drug_tests_df = cls._get_time_between_tests(drug_tests_df)
        print("Calculating time since last negative test...")
        drug_tests_df = cls._get_time_since_last_negative(drug_tests_df)
        print("Calculating time since and until positive test...")
        drug_tests_df = cls._get_time_since_and_until_positive(drug_tests_df)
        print("Calculating number of negatives since last positive...")
        drug_tests_df = cls._get_num_negatives_since_last_positive(drug_tests_df)
        print("Calculating number of positives since last negative...")
        drug_tests_df = cls._get_number_positives_since_last_negative(drug_tests_df)
        print("Calculating past drug test statistics...")
        drug_tests_df = cls._get_past_drug_test_stats(drug_tests_df)

        donor_df = cls.uppercase_col_names(donor_df)
        drug_tests_df = pd.merge(drug_tests_df, donor_df, on='DONOR_ID', how='left')

        return drug_tests_df

    @classmethod
    def uppercase_col_names(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all column names to uppercase.
        """
        df.columns = [col.upper() for col in df.columns]
        return df

    @classmethod
    def _calculate_statistics_per_test_day(cls, drug_tests_df: pd.DataFrame):
        """
        Calculate statistics for each donor based on positive drug test days.
        """
        print("Grouping drug tests by DONOR_ID, COLLECTION_ID, and DRUG_CLASS...")
        first_group = (
            drug_tests_df.groupby(['DONOR_ID', 'TIME', 'DRUG_CLASS'], as_index=False)
            .agg({
                'COLLECTION_ID': 'first',
                'DRUG_TEST_POSITIVE': 'any',
                'SHOWEDUP': 'any',
                'PROGRAM_TYPE': 'first'
            })
        )

        print("Aggregating statistics per test day...")
        # Get only positive drug tests
        positives = first_group[first_group['DRUG_TEST_POSITIVE']]
        # Aggregate the positive drug classes per test day
        pos_drugs = positives.groupby(['DONOR_ID', 'TIME'])['DRUG_CLASS'].agg(list)
        num_drugs_pos = positives.groupby(['DONOR_ID', 'TIME'])['DRUG_CLASS'].nunique()

        # Standard aggregations per test day
        second_group = (
            first_group.groupby(['DONOR_ID', 'TIME'], as_index=False)
            .agg(
                COLLECTION_ID=('COLLECTION_ID', 'first'),
                DRUG_POSITIVE=('DRUG_TEST_POSITIVE', 'any'),
                SHOWEDUP=('SHOWEDUP', 'any'),
                PROGRAM_TYPE=('PROGRAM_TYPE', 'first'),
                NUM_DRUGS_TESTED=('DRUG_CLASS', 'nunique'),
                DRUGS_TESTED=('DRUG_CLASS', 'unique'),
            )
        )

        # Merge the pre-computed positive drugs information
        second_group = second_group.merge(
            pos_drugs.rename('POSITIVE_DRUGS'), on=['DONOR_ID', 'TIME'], how='left'
        )
        second_group = second_group.merge(
            num_drugs_pos.rename('NUM_DRUGS_POSITIVE'), on=['DONOR_ID', 'TIME'], how='left'
        )

        # Fill NaNs for tests with no positive drugs
        second_group['POSITIVE_DRUGS'] = second_group['POSITIVE_DRUGS'].apply(lambda x: x if isinstance(x, list) else [])
        second_group['NUM_DRUGS_POSITIVE'] = second_group['NUM_DRUGS_POSITIVE'].fillna(0).astype(int)

        return second_group

    @classmethod
    def _get_participation_days(cls, drug_tests_df: pd.DataFrame):
        """
        Get the number of days since the first test for each donor.
        """
        drug_tests_df['FIRST_TEST_TIME'] = drug_tests_df.groupby('DONOR_ID')['TIME'].transform('min')
        drug_tests_df['PARTICIPATION_DAYS'] = (drug_tests_df['TIME'] - drug_tests_df['FIRST_TEST_TIME']).dt.days
        return drug_tests_df
    
    @classmethod
    def _get_time_between_tests(cls, drug_tests_df: pd.DataFrame):
        """
        Calculate the time between tests for each donor. It assumes that the 'TIME' column is in datetime format and
        that the DataFrame is sorted by 'DONOR_ID' and 'TIME'.
        Adds two new columns:
            - 'TIME_SINCE_LAST_TEST': Time in days since the last test for each donor.
            - 'TIME_UNTIL_NEXT_TEST': Time in days until the next test for each donor.
        The time is calculated as the difference between the current test time and the previous/next test time.
        If there is no previous/next test, the value will be NaN.
        The time is represented in days as a float.
        """        
        # Calculate the time difference between consecutive tests
        drug_tests_df['PREV_TEST_TIME'] = drug_tests_df.groupby('DONOR_ID')['TIME'].shift(1)
        drug_tests_df['NEXT_TEST_TIME'] = drug_tests_df.groupby('DONOR_ID')['TIME'].shift(-1)
        
        drug_tests_df['TIME_SINCE_LAST_TEST'] = (drug_tests_df['TIME'] - drug_tests_df['PREV_TEST_TIME']).dt.days.astype('float64')
        drug_tests_df['TIME_UNTIL_NEXT_TEST'] = (drug_tests_df['NEXT_TEST_TIME'] - drug_tests_df['TIME']).dt.days.astype('float64')

        return drug_tests_df
    
    @classmethod
    def _get_time_since_last_negative(cls, drug_tests_df: pd.DataFrame):
        """
        Calculate the time since the last negative test for each donor.
        """       
        def compute_within_group(group):
            # Mark negative test times, else NaN
            negative_times = group['TIME'].where(group['DRUG_POSITIVE'] == 0)
            
            # Time since last negative (look backward)
            last_negative_time = negative_times.ffill().shift(1)
            time_since = group['TIME'] - last_negative_time
            time_since[last_negative_time.isna()] = None
            
            group['TIME_SINCE_LAST_NEGATIVE'] = time_since.dt.days.astype('float64')
            return group

        # Apply within each donor
        result_df = (
            drug_tests_df
            .groupby('DONOR_ID', group_keys=False)
            .apply(compute_within_group)
            .reset_index(drop=True)
        )

        return result_df

    @classmethod
    def _get_time_since_and_until_positive(cls, drug_tests_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds two columns:
            - 'TIME_SINCE_LAST_POSITIVE'
            - 'TIME_UNTIL_NEXT_POSITIVE'
        for each test of each donor.
        """
        def compute_within_group(group):
            # Mark positive test times, else NaN
            positive_times = group['TIME'].where(group['DRUG_POSITIVE'] == 1)
            
            # Time since last positive (look backward)
            last_positive_time = positive_times.ffill().shift(1)
            time_since = group['TIME'] - last_positive_time
            time_since[last_positive_time.isna()] = None
            
            # Time until next positive (look forward)
            next_positive_time = positive_times.bfill().shift(-1)
            time_until = next_positive_time - group['TIME']
            time_until[next_positive_time.isna()] = None
            
            # Assign to DataFrame
            group['TIME_SINCE_LAST_POSITIVE'] = time_since
            group['TIME_UNTIL_NEXT_POSITIVE'] = time_until
            return group

        # Apply within each donor
        result_df = (
            drug_tests_df
            .groupby('DONOR_ID', group_keys=False)
            .apply(compute_within_group)
            .reset_index(drop=True)
        )

        result_df['TIME_SINCE_LAST_POSITIVE'] = result_df['TIME_SINCE_LAST_POSITIVE'].dt.days.astype('float64')
        result_df['TIME_UNTIL_NEXT_POSITIVE'] = result_df['TIME_UNTIL_NEXT_POSITIVE'].dt.days.astype('float64')

        return result_df
    
    @classmethod
    def _get_num_negatives_since_last_positive(cls, drug_tests_df: pd.DataFrame):
        def compute_within_group(group):
            test_number = np.arange(len(group))
            pos_idx = group['DRUG_POSITIVE'] == 1
            last_pos_test_number = pd.Series(test_number, index=group.index).where(pos_idx)
            last_pos_test_number = last_pos_test_number.ffill().shift(1)
            out = test_number - last_pos_test_number.values - 1  # <- Subtract 1 here for 0-indexing
            out[last_pos_test_number.isna()] = np.nan
            # For negative tests, they just keep counting (already correct)
            # For positive, still use this (it will be correct)
            return pd.Series(out, index=group.index)

        drug_tests_df['NUM_NEGATIVES_SINCE_LAST_POSITIVE'] = (
            drug_tests_df.groupby('DONOR_ID', group_keys=False)
            .apply(compute_within_group)
            .reset_index(level=0, drop=True)
        )

        return drug_tests_df

    @classmethod
    def _get_number_positives_since_last_negative(cls, drug_tests_df: pd.DataFrame):
        """
        For each donor, counts the length of the run of 1s immediately before each 0,
        and writes that value for the negative test that ends the run.
        Consecutive negatives after positives get 0.
        Positives and negatives not immediately following positives get NaN.
        """

        def run_length_for_negatives(group):
            values = group['DRUG_POSITIVE'].values
            out = np.full(len(values), np.nan)
            run_length = 0
            for i, v in enumerate(values):
                if v == 1:
                    run_length += 1
                else:
                    if run_length > 0:
                        out[i] = run_length
                        run_length = 0
                    else:
                        out[i] = 0
            return pd.Series(out, index=group.index)

        drug_tests_df['NUM_POSITIVES_SINCE_LAST_NEGATIVE'] = (
            drug_tests_df.groupby('DONOR_ID', group_keys=False).apply(run_length_for_negatives).reset_index(level=0, drop=True)
        )
        return drug_tests_df

    @classmethod
    def _get_past_drug_test_stats(cls, drug_tests_df: pd.DataFrame):
        stats_df = calculate_past_rolling_stats(drug_tests_df, group_col='DONOR_ID', id_col='COLLECTION_ID',
                                                    feature='DRUG_POSITIVE',
                                                     stats=['sum', 'mean'])
        stats_df = stats_df.drop(columns=['DONOR_ID'])
        
        drug_tests_df = drug_tests_df.merge(stats_df, on='COLLECTION_ID', how='left')

        return drug_tests_df

    @classmethod
    def _identify_positive_periods(cls, df: pd.DataFrame):
        df = df.sort_values(['donor_id', 'time'])
        df['is_negative'] = df['positive_date'] == 0

        # Add cumulative count of negative results per donor
        df['neg_cumsum'] = df['is_negative'].groupby(df['donor_id']).cumsum()

        # Extract positive rows
        positives = df[df['positive_date'] == 1].copy()

        # Get last positive time per donor
        positives['last_positive_date'] = positives.groupby('donor_id')['time'].shift(1)

        # Get negative count at current time and at last positive time
        df_for_merge = df[['donor_id', 'time', 'neg_cumsum']]

        # Join "time with last positive date"
        positives = positives.merge(
            df_for_merge.rename(columns={'time': 'last_positive_date', 'neg_cumsum': 'neg_count_at_last_pos'}),
            on=['donor_id', 'last_positive_date'],
            how='left'
        )
        # Join "time with time"
        positives = positives.merge(
            df_for_merge.rename(columns={'time': 'time', 'neg_cumsum': 'neg_count_at_current'}),
            on=['donor_id', 'time'],
            how='left'
        )

        # Calculate the number of negative tests since the last positive
        positives['negatives_since_last_positive'] = (
            positives['neg_count_at_current'] - positives['neg_count_at_last_pos'].fillna(0)
        ).astype('Int64')

        negatives = df[df['is_negative']].copy()
        negatives = negatives[['donor_id', 'time']].rename(columns={'time': 'last_negative_date'})

        # Use merge_asof to find the last negative date before each positive test
        # This is similar to a left-join except that we match on nearest key rather than equal keys. Both DataFrames must be sorted by the key.
        positives = pd.merge_asof(
            positives.sort_values('time'),
            negatives.sort_values('last_negative_date'),
            by='donor_id',
            left_on='time',
            right_on='last_negative_date',
            direction='backward'
        )
        final_columns = [
            'donor_id', 'time', 'showed_up', 'last_positive_date', 'last_negative_date',
            'negatives_since_last_positive', 'num_drug_classes_tested', 'positive_drug_classes', 'num_drug_classes_positive'
        ]
        return positives[final_columns].sort_values(['donor_id', 'time'])

