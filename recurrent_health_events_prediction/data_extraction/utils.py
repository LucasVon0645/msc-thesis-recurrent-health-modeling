from recurrent_health_events_prediction.data_extraction.data_types import DiseaseType

def assign_charlson_category(df, icd_column="ICD9_CODE"):
    """
    This function assigns to ICD-9 codes the categories used by
    the Charlson Comorbidity Index (CCI)
    --
    Reference for CCI:
    (1) Charlson ME, Pompei P, Ales KL, MacKenzie CR. (1987) A new method
    of classifying prognostic comorbidity in longitudinal studies: 
    development and validation.J Chronic Dis; 40(5):373-83.
    --
    (2) Charlson M, Szatrowski TP, Peterson J, Gold J. (1994) Validation
    of a combined comorbidity index. J Clin Epidemiol; 47(11):1245-51.

    Reference for ICD-9-CM Coding Algorithms for Charlson
    Comorbidities:
    (3) Quan H, Sundararajan V, Halfon P, et al. Coding algorithms for
    defining Comorbidities in ICD-9-CM and ICD-10 administrative data.
    Med Care. 2005 Nov; 43(11): 1130-9.

    :param df: DataFrame containing ICD-9 codes
    :param icd_column: Name of the column containing ICD-9 codes
    :return: DataFrame with an additional column 'COMORBIDITY' containing the disease categories
    """
    def categorize(icd9):
        if icd9.startswith(("410", "412")):
            return DiseaseType.MYOCARDIAL_INFARCT.value
        elif icd9.startswith("428") or icd9 in ["39891", "40201", "40211", "40291", "40401", "40403", "40411", "40413", "40491", "40493"] or "4254" <= icd9[:4] <= "4259":
            return DiseaseType.CONGESTIVE_HEART_FAILURE.value
        elif icd9.startswith(("440", "441")) or icd9 in ["0930", "4373", "4471", "5571", "5579", "V434"] or "4431" <= icd9[:4] <= "4439":
            return DiseaseType.PERIPHERAL_VASCULAR_DISEASE.value
        elif "430" <= icd9[:3] <= "438" or icd9 == "36234":
            return DiseaseType.CEREBROVASCULAR_DISEASE.value
        elif icd9.startswith("290") or icd9 in ["2941", "3312"]:
            return DiseaseType.DEMENTIA.value
        elif "490" <= icd9[:3] <= "505" or icd9 in ["4168", "4169", "5064", "5081", "5088"]:
            return DiseaseType.CHRONIC_PULMONARY_DISEASE.value
        elif icd9.startswith("725") or icd9 in ["4465", "7100", "7101", "7102", "7103", "7104", "7140", "7141", "7142", "7148"]:
            return DiseaseType.RHEUMATIC_DISEASE.value
        elif icd9.startswith(("531", "532", "533", "534")):
            return DiseaseType.PEPTIC_ULCER_DISEASE.value
        elif icd9.startswith(("570", "571")) or icd9 in ["0706", "0709", "5733", "5734", "5738", "5739", "V427", "07022", "07023", "07032", "07033", "07044", "07054"]:
            return DiseaseType.MILD_LIVER_DISEASE.value
        elif icd9[:4] in ["2500", "2501", "2502", "2503", "2508", "2509"]:
            return DiseaseType.DIABETES_WITHOUT_COMPLICATION.value
        elif icd9[:4] in ["2504", "2505", "2506", "2507"]:
            return DiseaseType.DIABETES_WITH_COMPLICATION.value
        elif icd9[:3] in ["342", "343"] or icd9[:4] in ["3341", "3440", "3441", "3442", "3443", "3444", "3445", "3446", "3449"]:
            return DiseaseType.PARAPLEGIA.value
        elif icd9[:3] in ["582", "585", "586", "V56"] or icd9[:4] in ["5880", "V420", "V451"] or "5830" <= icd9[:4] <= "5837" or icd9[:5] in ["40301", "40311", "40391", "40402", "40403", "40412", "40413", "40492", "40493"]:
            return DiseaseType.RENAL_DISEASE.value
        elif "140" <= icd9[:3] <= "172" or "1740" <= icd9[:4] <= "1958" or "200" <= icd9[:3] <= "208" or icd9 == "2386":
            return DiseaseType.MALIGNANT_CANCER.value
        elif icd9[:4] in ["4560", "4561", "4562"] or "5722" <= icd9[:4] <= "5728":
            return DiseaseType.SEVERE_LIVER_DISEASE.value
        elif icd9[:3] in ["196", "197", "198", "199"]:
            return DiseaseType.METASTATIC_SOLID_TUMOR.value
        elif icd9[:3] in ["042", "043", "044"]:
            return DiseaseType.AIDS.value
        else:
            return DiseaseType.OTHER.value  # For any other codes not classified

    df["COMORBIDITY"] = df[icd_column].apply(categorize)
    return df
