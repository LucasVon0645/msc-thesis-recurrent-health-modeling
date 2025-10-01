import os
from typing import Optional
import pandas as pd
import numpy as np
from recurrent_health_events_prediction.preprocessing.feature_extraction import FeatureExtractorDrugRelapse, FeatureExtractorMIMIC
from recurrent_health_events_prediction.preprocessing.utils import bin_time_col_into_cat, calculate_past_rolling_stats

class DataPreprocessorMIMIC:
    def __init__(self, config):
        self.config = config
        self.admission_types_to_consider = config.get("admission_types_to_consider")

    def preprocess(self, **kwargs):
        print("Preprocessing MIMIC dataset...")
        print("Building features...")
        events_df = FeatureExtractorMIMIC.build_features(**kwargs)

        events_df = self._clip_features(events_df)
        print("Defining readmission events...")
        events_df = self._define_events(events_df)
        print("Filtering events...")
        events_df = self._filter_events(events_df)
        print("Adding log columns...")
        events_df = self._add_log_cols(events_df)

        # Define the last events for each patient
        print("Defining last events...")
        events_df = self._define_last_events(events_df)

        # Define historical past events for each patient
        print("Defining historical past events...")
        events_df = self._define_historical_past_events(events_df)

        print("Categorizing readmission time...")
        events_df = self._categorize_readmission_time(events_df)

        # Reorder columns for better organization
        events_df = self._reorder_columns(events_df)
        # Split the DataFrame into last events and historical events
        print("Splitting last and historical events...")
        historical_events_df, last_events_df = self._split_last_and_historical_events(events_df)

        # Save the processed data
        self.save_training_data(historical_events_df, last_events_df, events_df, self.config["preprocessed_path"])

    def _define_last_events_old(self, events_df: pd.DataFrame):
        """
        Define the last events for each patient in the dataset.
        This method identifies the last event for each patient based on the admission time and whether the patient had a hospital death event.
        It marks the last event with a new column 'LAST_EVENT' and returns the modified DataFrame.
        Args:
            events_df (pd.DataFrame): DataFrame containing event data with columns 'SUBJECT_ID', 'ADMITTIME', 'HADM_ID', and 'IN_HOSP_DEATH_EVENT'.
        Returns:
            pd.DataFrame: DataFrame with an additional column 'LAST_EVENT' indicating the last event for each patient.
        """
        consider_death_after_discharge_gt = self.config.get("consider_death_after_discharge_gt", np.inf)
        events_df['IS_LAST_EVENT'] = 0

        invalid_events = (events_df["IN_HOSP_DEATH_EVENT"] == 1) | (
            (events_df["AFTER_HOSP_DEATH_EVENT"] == 1)
            & (events_df["DEATH_TIME_AFTER_LAST_DISCHARGE"] <= consider_death_after_discharge_gt)
        )

        valid_events_df = events_df[~invalid_events].sort_values(by=["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)

        # Identify the last readmission event of each patient
        last_event_ids = valid_events_df.groupby("SUBJECT_ID").last().reset_index()["HADM_ID"].unique()

        # Mark the last events
        events_df.loc[events_df["HADM_ID"].isin(last_event_ids), "IS_LAST_EVENT"] = 1

        return events_df
    
    def _define_last_events(self, events_df: pd.DataFrame):
        """
        Identify and flag the last event (admission) for each patient, based on readmission history 
        and specific death-related conditions.

        Definition of "last event":
            - If a patient has **only one hospital admission**, it is marked as the last event **only if**:
                - The patient did **not** die during that admission (`IN_HOSP_DEATH_EVENT != 1`), AND
                - The patient either:
                    - Did **not** die after discharge (i.e., `DEATH_TIME_AFTER_LAST_DISCHARGE` is missing/NaN), OR
                    - Died **later** than a configured threshold (`DEATH_TIME_AFTER_LAST_DISCHARGE > consider_death_after_discharge_gt`).

            - If a patient has **multiple admissions**, the second-to-last (penultimate) admission is always marked 
            as the last event, regardless of death conditions.

        The function adds a new column `'IS_LAST_EVENT'` to the DataFrame:
            - 1 if the event is considered the patient's "last event" per above rules
            - 0 otherwise

        Args:
            events_df (pd.DataFrame): DataFrame containing hospital admission events with the following columns:
                - 'SUBJECT_ID': Patient identifier
                - 'ADMITTIME': Timestamp of hospital admission
                - 'HADM_ID': Unique hospital admission ID
                - 'IN_HOSP_DEATH_EVENT': 1 if patient died during this admission, else 0
                - 'DEATH_TIME_AFTER_LAST_DISCHARGE': Time (in days) to death after last discharge, or NaN if patient did not die

        Returns:
            pd.DataFrame: A copy of the input DataFrame with an additional 'IS_LAST_EVENT' column.
        """
        consider_death_after_discharge_gt = self.config.get("consider_death_after_discharge_gt", np.inf)
        events_df['IS_LAST_EVENT'] = 0

        # Sort events per patient
        events_df = events_df.sort_values(by=["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)

        # Count admissions per patient
        events_df["ADMISSION_COUNT"] = events_df.groupby("SUBJECT_ID")["HADM_ID"].transform("count")

        # Mask for patients with one admission
        single_adm_mask = events_df["ADMISSION_COUNT"] == 1

        # Apply exclusion rules for single admissions
        valid_single_adm_mask = (
            single_adm_mask &
            (events_df["IN_HOSP_DEATH_EVENT"] != 1) &
            (
                events_df["DEATH_TIME_AFTER_LAST_DISCHARGE"].isna() |
                (events_df["DEATH_TIME_AFTER_LAST_DISCHARGE"] > consider_death_after_discharge_gt)
            )
        )

        # Select valid single admission HADM_IDs
        single_valid_hadm_ids = events_df.loc[valid_single_adm_mask, "HADM_ID"]

        # For patients with multiple admissions, select the penultimate
        multi_adm_df = events_df[~single_adm_mask].copy()
        multi_adm_df["RANK"] = multi_adm_df.groupby("SUBJECT_ID").cumcount(ascending=True)
        penultimate_df = multi_adm_df.groupby("SUBJECT_ID")["RANK"].transform("max") - 1
        penultimate_mask = multi_adm_df["RANK"] == penultimate_df
        penultimate_hadm_ids = multi_adm_df.loc[penultimate_mask, "HADM_ID"]

        # Combine valid HADM_IDs and mark
        valid_last_hadm_ids = pd.concat([single_valid_hadm_ids, penultimate_hadm_ids])
        events_df.loc[events_df["HADM_ID"].isin(valid_last_hadm_ids), "IS_LAST_EVENT"] = 1

        # Drop helper columns
        events_df = events_df.drop(columns=["ADMISSION_COUNT"], errors='ignore')
        # Sort the DataFrame by SUBJECT_ID and ADMITTIME again
        events_df = events_df.sort_values(by=["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)
        return events_df

    def _define_historical_past_events(self, events_df: pd.DataFrame):
        """
        Mark historical past events for each patient — i.e., all events occurring before the IS_LAST_EVENT.

        Assumes:
            - 'IS_LAST_EVENT' is already defined.
            - 'ADMITTIME' exists and is comparable.
            - Only one IS_LAST_EVENT per SUBJECT_ID.

        Args:
            events_df (pd.DataFrame): DataFrame containing at least 'SUBJECT_ID', 'ADMITTIME', and 'IS_LAST_EVENT'.

        Returns:
            pd.DataFrame: Modified DataFrame with 'IS_HISTORICAL_EVENT' column added.
        """
        # Find the ADMITTIME of the IS_LAST_EVENT for each SUBJECT_ID
        last_events = events_df.loc[events_df['IS_LAST_EVENT'] == 1, ['SUBJECT_ID', 'ADMITTIME']]
        last_events = last_events.rename(columns={'ADMITTIME': 'LAST_ADMITTIME'})

        # Merge the LAST_ADMITTIME into the original dataframe
        events_df = events_df.merge(last_events, on='SUBJECT_ID', how='left')

        # Mark as historical if ADMITTIME < LAST_ADMITTIME
        events_df['IS_HISTORICAL_EVENT'] = (events_df['ADMITTIME'] < events_df['LAST_ADMITTIME']).astype(int)

        # Drop temporary column
        events_df.drop(columns='LAST_ADMITTIME', inplace=True)

        return events_df

    def _clip_features(self, events_df: pd.DataFrame):
        """
        Clip the features in the events DataFrame based on the configuration.
        """
        clip_features_config = self.config.get("clip_features")

        if clip_features_config is not None:
            for feature, bounds in clip_features_config.items():
                events_df[feature] = events_df[feature].clip(lower=bounds.get("min"), upper=bounds.get("max"))

        return events_df

    def _filter_events(self, events_df: pd.DataFrame):

        min_age = self.config["filters"]["AGE"].get("min", 0)
        max_age = self.config["filters"]["AGE"].get("max", np.inf)

        age_mask = (events_df["AGE"] >= min_age) & (events_df["AGE"] <= max_age)
        events_df = events_df[age_mask]

        max_hospitalization_days = self.config["filters"]["HOSPITALIZATION_DAYS"].get("max", None)
        if max_hospitalization_days is not None:
            subjects_with_very_high_hospitalization_days = events_df[events_df["HOSPITALIZATION_DAYS"] > max_hospitalization_days]["SUBJECT_ID"].unique()
            events_df = events_df[~events_df["SUBJECT_ID"].isin(subjects_with_very_high_hospitalization_days)]

        # Filter out patients with events with negative or zero EVENT_DURATION
        discard_readmissions_or_censoring_with_invalid_time = events_df[(events_df["EVENT_DURATION"] <= 0) & (events_df["IN_HOSP_DEATH_EVENT"] != 1)]["SUBJECT_ID"].unique()
        events_df = events_df[~events_df["SUBJECT_ID"].isin(discard_readmissions_or_censoring_with_invalid_time)]

        # Filter out patients with readmission times less than minimum
        min_readmission_time = self.config["filters"]["READMISSION_TIME"].get("min", 0)
        if min_readmission_time > 0:
            discard_readmissions_with_invalid_time = events_df[events_df["DAYS_UNTIL_NEXT_HOSPITALIZATION"] < min_readmission_time]["SUBJECT_ID"].unique()
            events_df = events_df[~events_df["SUBJECT_ID"].isin(discard_readmissions_with_invalid_time)]

        # Filter by admission types if specified
        if self.admission_types_to_consider:
            events_df = events_df[events_df["ADMISSION_TYPE"].isin(self.admission_types_to_consider)]
            events_df = events_df[events_df["NEXT_ADMISSION_TYPE"].isin(self.admission_types_to_consider)]

        return events_df

    def _reorder_columns(self, events_df: pd.DataFrame):
        """
        Reorder the columns of the events DataFrame based on the configuration.
        """
        columns_order = events_df.columns.tolist()
        columns_order.remove("SUBJECT_ID")
        columns_order.remove("HADM_ID")
        columns_order.remove("ADMITTIME")
        columns_order.remove("DISCHTIME")
        columns_order.remove("ADMISSION_TYPE")
        columns_order = ["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "ADMISSION_TYPE"] + columns_order
        events_df = events_df[columns_order]
        return events_df

    def _define_events(self, events_df: pd.DataFrame):
        """
        Define the events for the disease recurrence prediction.
        """
        censoring_limit = self.config.get("censoring_limit", 120)
        # Uncensored events
        return_to_hosp_known_df = events_df[~events_df["NEXT_ADMITTIME"].isna()].copy()
        return_to_hosp_known_df["READMISSION_EVENT"] = 1  # Mark as uncensored. Days until next hospitalization higher than limit will be set to 0 later.
        return_to_hosp_known_df["EVENT_DURATION"] = return_to_hosp_known_df['DAYS_UNTIL_NEXT_HOSPITALIZATION']
        return_to_hosp_known_df["IN_HOSP_DEATH_EVENT"] = 0
        return_to_hosp_known_df["AFTER_HOSP_DEATH_EVENT"] = 0
        known_hosp_to_censor = (return_to_hosp_known_df["EVENT_DURATION"] > censoring_limit)
        return_to_hosp_known_df["EVENT_DURATION"] = np.where(known_hosp_to_censor, censoring_limit, return_to_hosp_known_df["EVENT_DURATION"])  # Cap at censoring limit
        return_to_hosp_known_df["READMISSION_EVENT"] = np.where(known_hosp_to_censor, 0, return_to_hosp_known_df["READMISSION_EVENT"])  # Mark as censored if capped

        # True Censored events. These are events where the next admission time is not available.
        # Censoring is defined as the time from discharge to death. Event duration will be None if no death occurred.
        no_next_admission_event_df = events_df[events_df["NEXT_ADMITTIME"].isna()].copy()
        no_next_admission_event_df["READMISSION_EVENT"] = 0
        no_next_admission_event_df["EVENT_DURATION"] = (no_next_admission_event_df["DOD"] - no_next_admission_event_df["DISCHTIME"]).dt.total_seconds() / (24 * 3600)  # Convert to days

        # Mark as hospital death if EVENT_DURATION is less than 1 day
        no_next_admission_event_df["IN_HOSP_DEATH_EVENT"] = np.where(no_next_admission_event_df["EVENT_DURATION"] < 1, 1, 0)
        no_next_admission_event_df["EVENT_DURATION"] = np.where(no_next_admission_event_df["IN_HOSP_DEATH_EVENT"], 0.0, no_next_admission_event_df["EVENT_DURATION"])

        # Mark as after hospital death if DOD is not NaN and DOD_HOSP is NaN
        no_next_admission_event_df["AFTER_HOSP_DEATH_EVENT"] = np.where(no_next_admission_event_df["EVENT_DURATION"] > 1, 1, 0)
        # Save true time until death after last discharge

        no_next_admission_event_df["IN_HOSP_DEATH_EVENT"] = no_next_admission_event_df["IN_HOSP_DEATH_EVENT"].astype(int)  # Convert to int
        no_next_admission_event_df["AFTER_HOSP_DEATH_EVENT"] = no_next_admission_event_df["AFTER_HOSP_DEATH_EVENT"].astype(int)  # Convert to int

        no_next_admission_event_df["EVENT_DURATION"] = no_next_admission_event_df["EVENT_DURATION"].clip(upper=censoring_limit)  # Cap at censoring limit, minimum 1 day
        no_next_admission_event_df["EVENT_DURATION"] = no_next_admission_event_df["EVENT_DURATION"].fillna(censoring_limit)

        # Append censored events to historical events
        events_df = pd.concat([return_to_hosp_known_df, no_next_admission_event_df], ignore_index=True)

        return events_df

    def _split_last_and_historical_events(self, events_df: pd.DataFrame):
        """
        Splits the events DataFrame into historical events and last events.
        Args:
            events_df (pd.DataFrame): DataFrame containing event data with a column 'IS_LAST_EVENT' indicating the last event for each patient.
        Returns:
            tuple: A tuple containing two DataFrames:
                - historical_events_df: DataFrame with historical events.
                - last_events_df: DataFrame with last events.
        """
        last_events_df = events_df[events_df['IS_LAST_EVENT'] == 1]
        historical_events_df = events_df[events_df['IS_HISTORICAL_EVENT'] == 1]

        return historical_events_df, last_events_df

    def save_training_data(self, historical_events_df, last_events_df, all_events_df, output_path):
        """
        Saves the processed historical and last events dataframes to the specified output path.
        Args:
            historical_events_df (pd.DataFrame): DataFrame with historical events.
            last_events_df (pd.DataFrame): DataFrame with last events.
            output_path (str): Path to save the processed data.
        """
        os.makedirs(output_path, exist_ok=True)
        print(f"Saving processed data to {output_path}")
        historical_events_df.to_csv(f"{output_path}/historical_events.csv", index=False)
        last_events_df.to_csv(f"{output_path}/last_events.csv", index=False)
        all_events_df.to_csv(f"{output_path}/all_events.csv", index=False)

    def _add_log_cols(self, events_df: pd.DataFrame):
        """
        Adds log columns to the events DataFrame.
        This method calculates the logarithm of certain columns in the DataFrame and adds them as new columns.
        Args:
            events_df (pd.DataFrame): DataFrame containing event data with columns to be transformed.
        Returns:
            pd.DataFrame: DataFrame with added log columns.
        """

        log_cols = self.config.get("log_cols_to_add", [])

        for col in log_cols:
            if col not in events_df.columns:
                print(f"Warning: Column '{col}' not found in DataFrame. Skipping log transformation for this feature.")
                continue
            events_df[f"LOG_{col}"] = np.log1p(events_df[col])

        return events_df

    def _categorize_readmission_time(self, admissions_df):
        """
        Categorize the readmission time into bins.
        """
        labels = self.config.get("event_duration_labels", ['0-30', '30-120', '120+'])
        bins = self.config.get("event_duration_bins", [0, 30, 120])

        print("Categories: ", labels)
        print("Bins: ", bins)

        admissions_df, _ = bin_time_col_into_cat(admissions_df, bins, labels,
                                                 cat_col_name="READMISSION_TIME_CAT",
                                                 col_to_bin="EVENT_DURATION",
                                                 apply_encoding=False)
        admissions_df['READMISSION_TIME_CAT_ENCODED'] = admissions_df['READMISSION_TIME_CAT'].cat.codes
        admissions_df['READMISSION_TIME_CAT_ENCODED'] = admissions_df['READMISSION_TIME_CAT_ENCODED'].fillna(-1).astype(int)
        admissions_df["READMISSION_TIME_CAT"] = np.where(
            (admissions_df["IS_LAST_EVENT"] == 0)
            & (admissions_df["IS_HISTORICAL_EVENT"] == 0),
            None,
            admissions_df["READMISSION_TIME_CAT"],
        )
        admissions_df["READMISSION_TIME_CAT_ENCODED"] = np.where(
            (admissions_df["IS_LAST_EVENT"] == 0)
            & (admissions_df["IS_HISTORICAL_EVENT"] == 0),
            -1,
            admissions_df["READMISSION_TIME_CAT_ENCODED"],
        )

        return admissions_df

class DataPreprocessorDrugRelapse:
    def __init__(self, config):
        self.config = config

    def preprocess(self, **kwargs):
        print("Preprocessing drug tests and relapse data...")
        # Extract drug tests and relapse data
        # This assumes that FeatureExtractorDrugRelapse has a method build_features that returns a DataFrame with drug tests and relapse information
        # The kwargs should contain necessary parameters for the feature extraction
        print("Building features for drug tests and relapse...")
        drug_tests_df = FeatureExtractorDrugRelapse.build_features(**kwargs)

        drug_tests_df = drug_tests_df.sort_values(by=["DONOR_ID", "TIME"]).reset_index(drop=True)

        print("Identifying relapse periods...")
        relapses_df = self._extract_relapse_periods(drug_tests_df)

        print("Getting number of previous relapses...")
        relapses_df = self._get_num_prev_relapses(relapses_df)

        print("Categorize relapse durations...")
        relapses_df = self._categorize_event_duration(relapses_df)
        relapses_df = self._binary_encode_event_duration(relapses_df)

        print("Adding to relapses information of drug tests...")
        relapses_df = self._get_info_positive_before_relapse_start(relapses_df, drug_tests_df)
        relapses_df = self._add_additional_cols_collection_relapse_start(relapses_df, drug_tests_df)

        print("Adding log columns to relapses and drug tests...")
        relapses_df = self._add_log_cols(relapses_df)
        drug_tests_df = self._add_log_cols(drug_tests_df)

        print("Calculating past relapse time statistics...")
        relapses_df = self._get_past_relapses_time_stats(relapses_df)

        print("Defining last events...")
        relapses_df = self._define_last_events(relapses_df)
    
        print("Defining historical drug tests and relapses...")
        drug_tests_df, relapses_df = self._define_historical_drug_tests_and_relapses(relapses_df, drug_tests_df)

        print("Splitting last and historical events...")
        last_relapses_df, historical_relapses_df, historical_drug_tests_df = self._split_last_and_historical_events(drug_tests_df, relapses_df)

        output_path = self.config["preprocessed_path"]

        print(f"Saving processed data to {output_path}")

        filename = "all_drug_tests.csv"
        self.save_training_data(drug_tests_df, filename, output_path)

        filename = "historical_drug_tests.csv"
        self.save_training_data(historical_drug_tests_df, filename, output_path)

        filename = "all_relapses.csv"
        self.save_training_data(relapses_df, filename, output_path)

        filename = "last_relapses.csv"
        self.save_training_data(last_relapses_df, filename, output_path)

        filename = "historical_relapses.csv"
        self.save_training_data(historical_relapses_df, filename, output_path)
    
    def _define_last_events(self, relapses_df: pd.DataFrame):
        """
        Define the last meaningful event for each donor in the relapse DataFrame.
        For each donor:
            - If there is a relapse event (RELAPSE_EVENT == 1) with NUM_PREV_RELAPSES > 0, 
            mark the last such event as IS_LAST_EVENT = 1.
            - Otherwise, mark the last available event (any type) as IS_LAST_EVENT = 1.
        
        Args:
            relapses_df (pd.DataFrame): DataFrame containing relapse data with 
                columns 'DONOR_ID', 'RELAPSE_EVENT', 'NUM_PREV_RELAPSES', 
                'RELAPSE_START', and 'COLLECTION_ID'.
        
        Returns:
            pd.DataFrame: DataFrame with an additional column 'IS_LAST_EVENT'.
        """
        relapses_df['IS_LAST_EVENT'] = 0

        # Sort to identify last events
        df_sorted = relapses_df.sort_values(['DONOR_ID', 'RELAPSE_START'])

        # Find valid relapse events with NUM_PREV_RELAPSES > 0
        valid_relapses = df_sorted[
            (df_sorted['RELAPSE_EVENT'] == 1) & 
            (df_sorted['NUM_PREV_RELAPSES'] > 0)
        ]

        # Last valid relapse per DONOR_ID
        last_valid_relapses = valid_relapses.groupby('DONOR_ID').tail(1)
        relapses_df.loc[last_valid_relapses.index, 'IS_LAST_EVENT'] = 1

        # For donors not covered above, fallback to last available row
        remaining_donors = set(relapses_df['DONOR_ID']) - set(last_valid_relapses['DONOR_ID'])
        fallback_rows = df_sorted[df_sorted['DONOR_ID'].isin(remaining_donors)].groupby('DONOR_ID').tail(1)
        relapses_df.loc[fallback_rows.index, 'IS_LAST_EVENT'] = 1

        return relapses_df

    def _define_historical_drug_tests_and_relapses(self, relapses_df: pd.DataFrame, drug_tests_df: pd.DataFrame):
        """
        Marks historical drug tests and relapses for each donor.

        For each donor, identifies their last relapse events (`IS_LAST_EVENT == 1`) in `relapses_df`
        and marks all drug tests in `drug_tests_df` as historical if they occurred before the
        start of this relapse (`RELAPSE_START`).

        Additionally, adds an `IS_HISTORICAL_EVENT` column to both dataframes:
        - In `drug_tests_df`, `IS_HISTORICAL_EVENT` is True if the drug test occurred before the donor's first `RELAPSE_START` among their last relapse events.
        - In `relapses_df`, `IS_HISTORICAL_EVENT` is 1 for all relapse events that occurred before the last one.

        Args:
            relapses_df (pd.DataFrame): Must include 'DONOR_ID', 'RELAPSE_START', 'IS_LAST_EVENT'.
            drug_tests_df (pd.DataFrame): Must include 'DONOR_ID' and 'TIME'.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Updated drug_tests_df and relapses_df
        """
        # Ensure datetime columns
        drug_tests_df = drug_tests_df.copy()
        relapses_df = relapses_df.copy()

        drug_tests_df['TIME'] = pd.to_datetime(drug_tests_df['TIME'])
        relapses_df['RELAPSE_START'] = pd.to_datetime(relapses_df['RELAPSE_START'])

        # ---- Drug tests ----
        last_relapses_df = relapses_df[relapses_df['IS_LAST_EVENT'] == 1]
        first_relapse_start = (
            last_relapses_df
            .groupby('DONOR_ID', as_index=False)['RELAPSE_START']
            .min()
            .rename(columns={'RELAPSE_START': 'FIRST_RELAPSE_START'})
        )

        drug_tests_df = pd.merge(drug_tests_df, first_relapse_start, on='DONOR_ID', how='left')

        drug_tests_df['IS_HISTORICAL_EVENT'] = (
            (drug_tests_df['TIME'] < drug_tests_df['FIRST_RELAPSE_START']) &
            drug_tests_df['FIRST_RELAPSE_START'].notna()
        )

        drug_tests_df.drop(columns=['FIRST_RELAPSE_START'], inplace=True)

        # ---- Relapses ----
        last_relapse_times = last_relapses_df.set_index('DONOR_ID')['RELAPSE_START']
        relapses_df['LAST_RELAPSE_START'] = relapses_df['DONOR_ID'].map(last_relapse_times)

        relapses_df['IS_HISTORICAL_EVENT'] = (
            relapses_df['RELAPSE_START'] < relapses_df['LAST_RELAPSE_START']
        ).astype(int)

        relapses_df.drop(columns=['LAST_RELAPSE_START'], inplace=True)

        return drug_tests_df, relapses_df

    def _split_last_and_historical_events(self, drug_tests_df: pd.DataFrame, relapses_df: pd.DataFrame):
        """
        Splits the events DataFrame into historical events and last events.
        Args:
            events_df (pd.DataFrame): DataFrame containing event data with a column 'IS_LAST_EVENT' indicating the last event for each patient.
        Returns:
            tuple: A tuple containing two DataFrames:
                - historical_events_df: DataFrame with historical events.
                - last_events_df: DataFrame with last events.
        """
        last_relapses_df = relapses_df[relapses_df['IS_LAST_EVENT'] == 1]
        historical_relapses_df = relapses_df[relapses_df['IS_HISTORICAL_EVENT'] == 1]

        historical_drug_tests_df = drug_tests_df[drug_tests_df['IS_HISTORICAL_EVENT'] == 1]
        
        return last_relapses_df, historical_relapses_df, historical_drug_tests_df

    def _get_info_positive_before_relapse_start(self, relapses_df: pd.DataFrame, drug_tests_df: pd.DataFrame):
        """
        Get information about the last positive drug test before the relapse start.
        This function merges the drug test information into the relapses DataFrame.
        It adds the following columns to the relapses DataFrame:
        - `PREV_POSITIVE_DRUGS`: The drugs that were positive in the last positive drug test before the relapse start.
        - `PREV_NUM_DRUGS_POSITIVE`: The number of drugs that were positive in the last positive drug test before the relapse start.
        Args:
            relapses_df (pd.DataFrame): DataFrame containing relapse events with column 'PREV_POS_COLLECTION_ID'.
            drug_tests_df (pd.DataFrame): DataFrame containing drug test events with columns 'COLLECTION_ID', 'POSITIVE_DRUGS', and 'NUM_DRUGS_POSITIVE'.
        Returns:
            pd.DataFrame: The relapses DataFrame with additional columns for the last positive drug test information.
        """
        drug_test_info_df = drug_tests_df[['COLLECTION_ID', 'POSITIVE_DRUGS', 'NUM_DRUGS_POSITIVE']]
        drug_test_info_df = drug_test_info_df.rename(columns={'COLLECTION_ID': 'PREV_POS_COLLECTION_ID',
                                                              'POSITIVE_DRUGS': 'PREV_POSITIVE_DRUGS',
                                                              'NUM_DRUGS_POSITIVE': 'PREV_NUM_DRUGS_POSITIVE'})
        relapses_df = pd.merge(
            relapses_df,
            drug_test_info_df,
            on='PREV_POS_COLLECTION_ID',
            how='left',
        )

        relapses_df.drop(columns=['PREV_POS_COLLECTION_ID'], inplace=True)

        return relapses_df

    def _add_additional_cols_collection_relapse_start(self, relapses_df: pd.DataFrame, drug_tests_df: pd.DataFrame):
        columns_to_add = self.config.get("cols_to_add_start_relapse", ["DRUG_POSITIVE_PAST_MEAN", "PROGRAM_TYPE"])
        drug_tests_info_df = drug_tests_df[['COLLECTION_ID'] + columns_to_add]

        relapses_df = pd.merge(
            relapses_df,
            drug_tests_info_df,
            on='COLLECTION_ID',
            how='left'
        )
        
        return relapses_df

    def _extract_relapse_periods(self, drug_tests_df: pd.DataFrame):
        """
        Extract relapse periods from the drug tests DataFrame.
        A relapse period is defined as a sequence of negative tests after a positive test.
        Each period is defined by its start and end times, the duration of the event, and whether it was a relapse event.

        The function returns a DataFrame with the following columns:
        - `DONOR_ID`: ID of the donor
        - `COLLECTION_ID`: ID of the collection
        - `RELAPSE_START`: Start time of the relapse period
        - `RELAPSE_END`: End time of the relapse period
        - `EVENT_DURATION`: Duration of the relapse period
        - `RELAPSE_EVENT`: 1 if it was a relapse event, 0 if it was a relapse-free period
        - `NUM_TESTS_PERIOD`: Number of tests in the relapse period
        """
        def patient_relabels(group):
            # Ensure sorted by time
            group = group.reset_index(drop=True)
            is_pos = group['DRUG_POSITIVE'].values
            n = len(group)
            results = []
            
            # Find all indices of positive and negative tests
            pos_idx = np.where(is_pos == 1)[0]
            neg_idx = np.where(is_pos == 0)[0]
            
            # If there are no positives, treat all as censored
            if len(pos_idx) == 0 and len(neg_idx) > 0:
                start = group.loc[neg_idx[0], 'TIME']
                end = group.loc[neg_idx[-1], 'TIME']
                event_duration = float((end - start).days)
                if event_duration <= 0:
                    event_duration = float('nan')
                    end = pd.NaT

                results.append({
                    'DONOR_ID': group.loc[0, 'DONOR_ID'],
                    'COLLECTION_ID': group.loc[0, 'COLLECTION_ID'],
                    'PREV_POS_COLLECTION_ID': None,
                    'RELAPSE_START': start,
                    'RELAPSE_END': end,
                    'EVENT_DURATION': event_duration,
                    'RELAPSE_EVENT': 0,
                    'NUM_TESTS_PERIOD': len(neg_idx)
                })
                return pd.DataFrame(results)
            
            # Identify the start of each relapse-free period (first negative after a positive)
            transitions = np.where((is_pos[:-1] == 1) & (is_pos[1:] == 0))[0]
            for trans in transitions:
                start_idx = trans + 1  # first negative after positive
                # Find next positive after start_idx
                next_positives = np.where(is_pos[start_idx+1:] == 1)[0]
                if len(next_positives) > 0:
                    end_idx = start_idx + next_positives[0] + 1  # first positive after start_idx
                    relapse_event = 1
                    prev_pos_collection_id = group.loc[trans, 'COLLECTION_ID']
                    num_tests_period = end_idx - start_idx
                else:
                    end_idx = n - 1  # last test (censored)
                    relapse_event = 0
                    prev_pos_collection_id = group.loc[trans, 'COLLECTION_ID']
                    num_tests_period = end_idx - start_idx + 1

                if end_idx <= start_idx:
                    continue
                
                end = group.loc[end_idx, 'TIME']
                start = group.loc[start_idx, 'TIME']

                event_duration = float((end - start).days)
                if event_duration <= 0:
                    event_duration = float('nan')
                    end = pd.NaT

                results.append({
                    'DONOR_ID': group.loc[start_idx, 'DONOR_ID'],
                    'COLLECTION_ID': group.loc[start_idx, 'COLLECTION_ID'],
                    'PREV_POS_COLLECTION_ID': prev_pos_collection_id,
                    'RELAPSE_START': start,
                    'RELAPSE_END': end,
                    'EVENT_DURATION': event_duration,
                    'RELAPSE_EVENT': relapse_event,
                    'NUM_TESTS_PERIOD': num_tests_period,
                })
            return pd.DataFrame(results)
        
        relapse_periods_df = (
            drug_tests_df
            .groupby('DONOR_ID', group_keys=False)
            .apply(patient_relabels)
            .reset_index(drop=True)
        )
        return relapse_periods_df

    def _add_log_cols(self, events_df: pd.DataFrame):
        """
        Adds log columns to the events DataFrame.
        This method calculates the logarithm of certain columns in the DataFrame and adds them as new columns.
        Args:
            events_df (pd.DataFrame): DataFrame containing event data with columns to be transformed.
        Returns:
            pd.DataFrame: DataFrame with added log columns.
        """
        
        log_cols = self.config.get("log_cols_to_add", [])
        
        for col in log_cols:
            if col not in events_df.columns:
                print(f"Warning: Column '{col}' not found in DataFrame. Skipping log transformation for this feature.")
                continue
            events_df[f"LOG_{col}"] = np.log1p(events_df[col])
        
        return events_df

    def _get_num_prev_relapses(self, events_df: pd.DataFrame):
        """
        Adds a column 'NUM_PREV_RELAPSES' to the dataframe indicating how many prior relapse events 
        each donor had before the current relapse event.
        
        Parameters:
        events_df (pd.DataFrame): Input dataframe with required columns:
        - DONOR_ID
        - COLLECTION_ID
        - RELAPSE_START
        - RELAPSE_EVENT
        
        Returns:
        pd.DataFrame: DataFrame with 'NUM_PREV_RELAPSES' column
        """
        # Filter for relapse events
        relapses = events_df[events_df['RELAPSE_EVENT'] == 1].copy()

        # Sort by DONOR_ID and RELAPSE_START to ensure chronological order
        relapses.sort_values(['DONOR_ID', 'RELAPSE_START'], inplace=True)

        # Count previous relapses for each donor
        relapses['NUM_PREV_RELAPSES'] = relapses.groupby('DONOR_ID').cumcount()

        # Merge back into the original dataframe using COLLECTION_ID
        events_df = events_df.merge(
            relapses[['COLLECTION_ID', 'NUM_PREV_RELAPSES']],
            on='COLLECTION_ID',
            how='left'
        )

        # Fill NaN for non-relapse periods with 0
        events_df['NUM_PREV_RELAPSES'] = events_df['NUM_PREV_RELAPSES'].fillna(0).astype(int)

        return events_df
    
    def _get_past_relapses_time_stats(self, relapses_df: pd.DataFrame):
        binary_event_col = self.config.get("binary_event_col", "EVENT_DURATION_BINARY")

        relapses_stats_df = calculate_past_rolling_stats(relapses_df,
                                     group_col='DONOR_ID',
                                     id_col='COLLECTION_ID',
                                     feature='LOG_TIME_UNTIL_NEXT_POSITIVE',
                                     prefix='LOG_TIME_RELAPSE',
                                     stats=['median', 'mean', 'std'])
                
        relapses_stats_df = relapses_stats_df.drop(columns=['DONOR_ID'])
        relapses_df = relapses_df.merge(relapses_stats_df, on='COLLECTION_ID', how='left')
        binary_col_shifted = f'PREV_{binary_event_col}'
        relapses_df[binary_col_shifted] = (
            relapses_df
            .groupby("DONOR_ID")[binary_event_col]
            .shift(1)
            .fillna(0)
            .astype(int)
        )

        relapses_stats_df = calculate_past_rolling_stats(relapses_df,
                                     group_col='DONOR_ID',
                                     id_col='COLLECTION_ID',
                                     feature=binary_event_col,
                                     stats=['mean', 'sum'])
        relapses_stats_df = relapses_stats_df.drop(columns=['DONOR_ID'])
        relapses_df = relapses_df.merge(relapses_stats_df, on='COLLECTION_ID', how='left')

        return relapses_df

    def _categorize_event_duration(self, events_df: float):
        """
        Categorizes the event duration into bins.
        This method creates a new column 'EVENT_DURATION_CATEGORY' in the DataFrame based on the 'EVENT_DURATION' column.
        Args:
            events_df (pd.DataFrame): DataFrame containing event data with a column 'EVENT_DURATION'.
        Returns:
            pd.DataFrame: DataFrame with an additional column 'EVENT_DURATION_CATEGORY'.
        """
        bins = self.config.get("event_duration_bins", [0, 30, 120])
        labels = self.config.get("event_duration_labels", ['0-30', '30-120', '120+'])
        
        events_df, _ = bin_time_col_into_cat(events_df, bins=bins, labels=labels, col_to_bin='EVENT_DURATION', cat_col_name='RELAPSE_DURATION_CATEGORY', apply_encoding=False)

        events_df['RELAPSE_DURATION_CATEGORY_ENCODED'] = events_df['RELAPSE_DURATION_CATEGORY'].cat.codes
        events_df['RELAPSE_DURATION_CATEGORY_ENCODED'] = events_df['RELAPSE_DURATION_CATEGORY_ENCODED'].fillna(-1).astype(int)
        
        return events_df

    def _binary_encode_event_duration(self, events_df: pd.DataFrame):
        """
        Binary encodes the event duration into a new column.
        This method creates a new column 'EVENT_DURATION_BINARY' in the DataFrame based on the 'EVENT_DURATION' column.
        Args:
            events_df (pd.DataFrame): DataFrame containing event data with a column 'EVENT_DURATION'.
        Returns:
            pd.DataFrame: DataFrame with an additional column 'EVENT_DURATION_BINARY'.
        """
        binary_threshold = float(self.config.get("binary_threshold", 90))
        binary_event_col = self.config.get("binary_event_col", "EVENT_DURATION_BINARY")

        events_df[binary_event_col] = np.where(events_df['EVENT_DURATION'] < binary_threshold, 1, 0)

        return events_df

    def save_training_data(self, df, filename: str, output_path):
        """
        Saves the processed drug relapse data to the specified output path.
        Args:
            df (pd.DataFrame): DataFrame containing processed drug relapse data.
            output_path (str): Path to save the processed data.
        """
        os.makedirs(output_path, exist_ok=True)

        filepath = os.path.join(output_path, filename)

        print(f"Saving processed data to {filepath}")
        df.to_csv(filepath, index=False)
        print("Data saved successfully.")
