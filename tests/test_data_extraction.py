import pandas as pd
from recurrent_health_events_prediction.data_extraction.utils import assign_charlson_category
from recurrent_health_events_prediction.data_extraction.data_types import DiseaseType

def test_charlson_category_assignment():
    data = {
        "ICD9_CODE": [
            "42822",  # Congestive heart failure
            "25040",  # Diabetes with complication
            "5733",   # Mild liver disease
            "042",    # AIDS
            "41090",  # Myocardial infarction
            "1971",   # Metastatic solid tumor
            "99999",  # Unknown/other
        ]
    }

    df = pd.DataFrame(data)
    result = assign_charlson_category(df)

    expected = [
        DiseaseType.CONGESTIVE_HEART_FAILURE.value,
        DiseaseType.DIABETES_WITH_COMPLICATION.value,
        DiseaseType.MILD_LIVER_DISEASE.value,
        DiseaseType.AIDS.value,
        DiseaseType.MYOCARDIAL_INFARCT.value,
        DiseaseType.METASTATIC_SOLID_TUMOR.value,
        DiseaseType.OTHER.value  # 99999 doesn't map to any Charlson category
    ]

    assert list(result["COMORBIDITY"]) == expected
