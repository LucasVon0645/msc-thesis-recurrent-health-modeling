import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import PowerTransformer

from recurrent_health_events_prediction.model.RecurrentHealthEventsHMM import (
    RecurrentHealthEventsHMM,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, np.nan],
            "b": [10.0, 20.0, 30.0, np.nan, 50.0],
            "c": [100, 200, 300, 400, 500],  # untouched column
        }
    )


def test_fit_transform_leaves_nans(sample_df):
    model = RecurrentHealthEventsHMM(
        {
            "power_transform_variables": ["a", "b"],
            "apply_power_transform": True,
            "n_states": 2,
            "features": {"a": "gaussian", "b": "gaussian", "c": "poisson"},
        }
    )

    df_transformed = model.fit_transform_power_variables(sample_df)

    # Check that NaNs are still NaNs
    assert pd.isna(df_transformed.loc[4, "a"])
    assert pd.isna(df_transformed.loc[3, "b"])


def test_transform_changes_non_nan_values_only(sample_df):
    model = RecurrentHealthEventsHMM(
        {
            "power_transform_variables": ["a", "b"],
            "apply_power_transform": True,
            "n_states": 2,
            "features": {"a": "gaussian", "b": "gaussian", "c": "poisson"},
        }
    )

    df_transformed_1 = model.fit_transform_power_variables(sample_df)
    df_transformed_2 = model.transform_with_fitted_power(sample_df)

    # Non-NaN transformed values should differ from original
    assert not np.allclose(
        sample_df["a"].dropna().values, df_transformed_2["a"].dropna().values
    )

    # Check that NaNs are still NaNs
    assert pd.isna(df_transformed_2.loc[4, "a"])
    assert pd.isna(df_transformed_2.loc[3, "b"])

    # Check if df_transformed_2 and df_transformed_1 are the same
    assert np.allclose(
        df_transformed_1["a"].dropna().values,
        df_transformed_2["a"].dropna().values
    )


def test_shape_and_index_preserved(sample_df):
    model = RecurrentHealthEventsHMM(
        {
            "power_transform_variables": ["a", "b"],
            "apply_power_transform": True,
            "n_states": 2,
            "features": {"a": "gaussian", "b": "gaussian", "c": "poisson"},
        }
    )

    _ = model.fit_transform_power_variables(sample_df)
    df_transform = model.transform_with_fitted_power(sample_df)

    # Shape and index must match
    assert df_transform.shape == sample_df.shape
    assert (df_transform.index == sample_df.index).all()


def test_no_crash_on_missing_column(sample_df):
    model = RecurrentHealthEventsHMM(
        {
            "power_transform_variables": ["a", "b"],
            "apply_power_transform": True,
            "n_states": 2,
            "features": {"a": "gaussian", "b": "gaussian", "c": "poisson"},
        }
    )
    model.power_transform_columns = ["a", "b", "nonexistent"]

    # Should not raise even though "nonexistent" is not in df
    model.fit_transform_power_variables(sample_df)
    model.transform_with_fitted_power(sample_df)


def test_untouched_column_remains_unchanged(sample_df):
    model = RecurrentHealthEventsHMM(
        {
            "power_transform_variables": ["a", "b"],
            "apply_power_transform": True,
            "n_states": 2,
            "features": {"a": "gaussian", "b": "gaussian", "c": "poisson"},
        }
    )

    _ = model.fit_transform_power_variables(sample_df)
    df_transform = model.transform_with_fitted_power(sample_df)

    # Column 'c' should remain unchanged
    assert np.array_equal(df_transform["c"], sample_df["c"])
