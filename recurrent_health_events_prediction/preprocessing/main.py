import yaml
from importlib import resources as impresources
from recurrent_health_events_prediction import configs

from recurrent_health_events_prediction.data_extraction.DataExtractor import DataExtractorMIMIC, DataExtractorDrugRelapse
from recurrent_health_events_prediction.data_extraction.data_types import DiseaseType

from recurrent_health_events_prediction.preprocessing.preprocessors import DataPreprocessorMIMIC, DataPreprocessorDrugRelapse

def main(dataset_name: str = "mimic"):
    with open((impresources.files(configs) / 'data_config.yaml')) as f:
        config = yaml.safe_load(f)

    dataset_config = config["dataset"][dataset_name]
    preprocessing_config = config["training_data"][dataset_name]

    if dataset_name == "mimic":
        selected_diseases = preprocessing_config.get("selected_diseases", [DiseaseType.CHRONIC_PULMONARY_DISEASE,
                                                            DiseaseType.CONGESTIVE_HEART_FAILURE,
                                                            DiseaseType.DIABETES_WITH_COMPLICATION,
                                                            DiseaseType.RENAL_DISEASE])
        selected_diseases = [DiseaseType(d) if isinstance(d, str) else d for d in selected_diseases]
        data_extractor = DataExtractorMIMIC(dataset_config, selected_diseases)

        data_extractor.load_data()
        admissions_df = data_extractor.get_admissions_df()
        icu_stays_df = data_extractor.get_icu_stays_df()
        procedures_df = data_extractor.get_procedures_df()
        prescriptions_df = data_extractor.get_prescriptions_df()
        patients_metadata_df = data_extractor.get_patients_df()

        preprocessor = DataPreprocessorMIMIC(preprocessing_config)
        
        preprocessor.preprocess(
            admissions_df=admissions_df,
            icu_stays_df=icu_stays_df,
            prescriptions_df=prescriptions_df,
            procedures_df=procedures_df,
            patients_metadata_df=patients_metadata_df,
        )
    
    elif dataset_name == "relapse":
        data_extractor = DataExtractorDrugRelapse(dataset_config)
        data_extractor.load_data()
        print("Data loaded successfully for relapse dataset.")

        donor_df = data_extractor.get_donor_df()
        drug_tests_df = data_extractor.get_drug_tests_df()

        preprocessor = DataPreprocessorDrugRelapse(preprocessing_config)
        preprocessor.preprocess(
            donor_df=donor_df,
            drug_tests_df=drug_tests_df
        )


if __name__ == "__main__":
    dataset_name = "mimic"  # relapse or mimic
    main(dataset_name)