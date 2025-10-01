import os
import pickle
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter
from lifelines import LogNormalAFTFitter
from lifelines.utils import concordance_index

from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import brier_score
from sksurv.metrics import cumulative_dynamic_auc

from sklearn.metrics import roc_auc_score

from recurrent_health_events_prediction.model.model_types import SurvivalModelType
from recurrent_health_events_prediction.model.utils import (
    get_median_and_ci,
    plot_auc,
    plot_survival_function_plotly,
)


class NextEventPredictionModel:
    def __init__(
        self, model_config: dict, model_type: Optional[SurvivalModelType] = None
    ):
        # Initialize the model with configuration parameters.
        self.model_params = model_config.get("model_params", {})
        if model_type is None:
            model_type = model_config.get("model_type", SurvivalModelType.COX_PH)
            self.model_type = (
                SurvivalModelType(model_type)
                if isinstance(model_type, str)
                else model_type
            )
        else:
            self.model_type = (
                model_type
                if isinstance(model_type, SurvivalModelType)
                else SurvivalModelType(model_type)
            )

        self.model_name: str = model_config.get("model_name", None)
        self.strata_col: str = model_config.get("strata_col", None)
        self.cluster_col: str = model_config.get("cluster_col", None)
        self.event_id_col: str = model_config.get("event_id_col", None)
        self.event_col: str = model_config.get("event_col", None)
        self.duration_col: str = model_config.get("duration_col", None)
        self.features = model_config.get("features", None)
        self.feature_names_in_: list[str] = self.features
        self.model_path: str = model_config.get("save_model_path", None)
        self.base_hmm_name: str = model_config.get("base_hmm_name", None)

        self.random_search_cv_results = None
        self.model_config = model_config

        # Initialize the model and scaler to None
        self.model: (
            KaplanMeierFitter
            | CoxPHFitter
            | WeibullAFTFitter
            | LogNormalAFTFitter
            | GradientBoostingSurvivalAnalysis
            | None
        ) = None
        self.scaler: StandardScaler | None = None

        # Model key performance metrics
        self.key_test_performance_metrics = {}
        self.key_train_performance_metrics = {}
        self.key_validation_performance_metrics = {}

    def scale_features(
        self,
        features_to_scale: list[str],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame = None,
    ):
        X_train = X_train.copy()
        if X_test is not None:
            X_test = X_test.copy()

        if len(features_to_scale) > 0:
            scaler = StandardScaler()
            X_train[features_to_scale] = scaler.fit_transform(
                X_train[features_to_scale]
            )
            if X_test is not None:
                X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])
            self.scaler = scaler
        else:
            raise ValueError("No features specified for scaling.")

        return X_train, X_test

    def fit(self, X: pd.DataFrame, model_type: Optional[SurvivalModelType] = None):
        if model_type is not None:
            self.set_model_type(model_type)

        event_col = self.event_col
        duration_col = self.duration_col

        # Check if event_col and duration_col are present in the DataFrame
        if (event_col not in X.columns) or (duration_col not in X.columns):
            raise ValueError(
                f"Columns '{event_col}' and '{duration_col}' must be present in the DataFrame."
            )

        cols = [duration_col, event_col] + self.features

        strata_col = self.strata_col
        if strata_col and (self.model_type != SurvivalModelType.COX_PH):
            print(
                "Warning: Strata column is only applicable for Cox Proportional Hazards based model. Strata will be ignored."
            )
            strata_col = None

        cluster_col = self.cluster_col
        if cluster_col and (self.model_type != SurvivalModelType.COX_PH):
            print(
                "Warning: Cluster column is only applicable for Cox Proportional Hazards based model. Cluster will be ignored."
            )
            cluster_col = None

        # Check if strata is in the DataFrame if specified
        if strata_col is not None:
            if strata_col not in X.columns:
                raise ValueError(
                    f"Strata column '{strata_col}' not found in the DataFrame."
                )
            cols = cols + [strata_col]

        # Check if cluster is in the DataFrame if specified
        if cluster_col is not None:
            if cluster_col not in X.columns:
                raise ValueError(
                    f"Cluster column '{cluster_col}' not found in the DataFrame."
                )
            cols = cols + [cluster_col]

        # Select only the relevant columns
        X = X[cols]

        if self.model_type == SurvivalModelType.COX_PH:
            self.model = CoxPHFitter(**self.model_params)
            self.model.fit(
                X,
                duration_col=duration_col,
                event_col=event_col,
                strata=strata_col,
                cluster_col=cluster_col,
                robust=cluster_col is not None,
            )
            self.key_train_performance_metrics["log_likelihood"] = (
                self.model.log_likelihood_
            )

        elif self.model_type == SurvivalModelType.KAPLAN_MEIER:
            self.model = KaplanMeierFitter(**self.model_params)
            self.model.fit(X[duration_col], event_observed=X[event_col])

        elif self.model_type == SurvivalModelType.LOGNORMAL_AFT:
            self.model = LogNormalAFTFitter(**self.model_params)
            self.model.fit(X, duration_col=duration_col, event_col=event_col)

        elif self.model_type == SurvivalModelType.WEIBULL_AFT:
            self.model = WeibullAFTFitter(**self.model_params)
            self.model.fit(X, duration_col=duration_col, event_col=event_col)

        elif self.model_type == SurvivalModelType.GBM:
            y_train = Surv.from_dataframe(event_col, duration_col, X)
            X_train = X.drop(columns=[duration_col, event_col])
            self.model = GradientBoostingSurvivalAnalysis(**self.model_params)
            self.model.fit(X_train, y_train)

        else:
            raise ValueError(f"Model type '{self.model_type}' is not supported.")

    def predict_median_with_ci_bounds(self, X_test, conditional_after=None, ci=0.95):
        """
        Predict the median survival time with confidence intervals.

        Args:
            X_test: DataFrame containing test features.
            ci: Confidence interval level (default is 0.95).

        Returns:
            A DataFrame with median survival times and confidence intervals.
        """
        cols = self.features
        strata_col = self.strata_col

        if strata_col is not None:
            cols = [strata_col] + cols

        surv_df = self.predict_survival(
            X_test, conditional_after
        )  # Ensure model is trained
        median_surv_df = get_median_and_ci(surv_df, ci=ci)

        return median_surv_df

    def predict_events_at_t(
        self, X_input: pd.DataFrame, t: float, conditional_after=None
    ):
        """
        Predict the survival probability at a specific time t for each event in the input DataFrame.

        Args:
            X_input: DataFrame containing test features.
            t: Time at which to compute the survival probability.
            conditional_after: Optional time after which to condition the survival function.

        Returns:
            A DataFrame with event IDs, true durations, true events, event status at time t and survival probabilities at time t.
            Columns: [event_id, 'time', 'true_duration', 'true_event', 'survival_prob_at_t', 'event_at_t'].
        """
        true_duration_col = self.duration_col
        true_event_col = self.event_col

        event_id_col = self.event_id_col

        if event_id_col in X_input.columns:
            X_input = X_input.set_index(event_id_col)

        surv_df = self.predict_survival(
            X_input, times=[t], conditional_after=conditional_after
        )

        if self.model_type == SurvivalModelType.KAPLAN_MEIER:
            survival_prob_at_t_list = np.repeat(surv_df.loc[t].values, len(X_input))
        else:
            survival_prob_at_t_list = surv_df.loc[t].values

        result_df = pd.DataFrame(
            {
                f"{self.event_id_col}": X_input.index.values,
                "time": [t] * len(X_input),
                "true_duration": X_input[true_duration_col].values,
                "true_event": X_input[true_event_col].values,
                "survival_prob_at_t": survival_prob_at_t_list,
                "prob_event_happened_before_t": 1 - survival_prob_at_t_list,
            }
        )

        def get_event_status_at_t(row, t):
            if row["true_event"] == 1 and row["true_duration"] <= t:
                return 1  # event happened before or at time t
            elif row["true_event"] == 0 and row["true_duration"] > t:
                return 0  # event did not happen before time t (still at risk). event censored but observation period is not over
            elif row["true_event"] == 0 and row["true_duration"] <= t:
                return (
                    np.nan
                )  # unknown event status at time t, t is greater than or equal to the observation period
            else:
                return 0  # if event happened after time t, we consider it as not happened at time t

        result_df["event_at_t"] = result_df.apply(
            lambda row: get_event_status_at_t(row, t), axis=1
        )

        return result_df

    def predict_survival(
        self, X_input, times: Optional[np.ndarray | list] = None, conditional_after=None
    ):
        """Predict survival functions for the test set.
        Args:
            X_input: DataFrame containing events' features.
            times: Optional array of times at which to compute survival probabilities.
            conditional_after: Optional time after which to condition the survival function.
        Returns:
            A DataFrame with survival functions indexed by time.
        """
        model = self.get_model()

        event_id_col = self.event_id_col
        if event_id_col in X_input.columns:
            X_input = X_input.set_index(event_id_col)

        if isinstance(model, KaplanMeierFitter):
            if times is None:
                surv_func = model.survival_function_
                return pd.DataFrame(surv_func)
            else:
                surv_func = model.predict(times)
                if isinstance(surv_func, pd.Series):
                    surv_func = pd.DataFrame(surv_func)
                else:
                    surv_func = pd.DataFrame(
                        [surv_func], index=times, columns=["KM_estimate"]
                    )
                return surv_func

        cols = self.features
        strata_col = self.strata_col

        if strata_col is not None:
            cols = [strata_col] + cols

        if isinstance(model, (CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter)):
            surv_functions = model.predict_survival_function(
                X_input[cols], times=times, conditional_after=conditional_after
            )

        elif isinstance(model, GradientBoostingSurvivalAnalysis):
            surv_functions = model.predict_survival_function(X_input[cols])
            if times is not None:
                # Extract survival probabilities
                surv_dict = {}
                for i, fn in enumerate(surv_functions):
                    survival_probabilities = fn(times)
                    surv_dict[i] = survival_probabilities
                # Convert to DataFrame
                surv_functions = pd.DataFrame(surv_dict, index=times)
            else:
                # Convert surv_funcs into a DataFrame
                surv_functions = pd.DataFrame(
                    {
                        func_idx: surv_functions[func_idx].y
                        for func_idx in range(len(surv_functions))
                    },  # Columns as y-values
                    index=surv_functions[0].x,  # times as x-values
                )

            surv_functions.columns = (
                X_input.index
            )  # Set the index of the DataFrame to match the test set indices

        return surv_functions

    def pred_partial_hazard(self, X_input: pd.DataFrame):
        model = self.get_model()
        if not isinstance(model, CoxPHFitter):
            raise ValueError(
                "Partial hazard prediction is only supported for CoxPHFitter models."
            )

        cols = self.features
        strata_col = self.strata_col

        event_id_col = self.event_id_col
        event_id_s = X_input[event_id_col] if event_id_col in X_input.columns else None

        if strata_col is not None:
            cols = [strata_col] + cols

        partial_hazards = self.model.predict_partial_hazard(X_input[cols])
        partial_hazard_df = pd.DataFrame(partial_hazards, columns=["partial_hazard"])
        if event_id_s is not None:
            partial_hazard_df[event_id_col] = event_id_s.values
        return partial_hazard_df

    def set_model_type(self, model_type: SurvivalModelType):
        if not isinstance(model_type, SurvivalModelType):
            raise ValueError(
                "model_type must be an instance of SurvivalModelType enum."
            )
        self.model_type = model_type

    def evaluate_c_index(self, X_input, save_metric=False, evaluation_set="test"):
        duration_col = self.duration_col
        event_col = self.event_col
        if duration_col not in X_input.columns or event_col not in X_input.columns:
            raise ValueError(
                f"Columns '{duration_col}' and '{event_col}' must be present in the test DataFrame."
            )

        if self.model is None:
            raise ValueError(
                "Model has not been trained yet. Call fit() before evaluate()."
            )

        strata_col = self.strata_col
        required_cols = [duration_col, event_col] + self.features
        if strata_col is not None:
            if strata_col not in X_input.columns:
                raise ValueError(
                    f"Strata column '{strata_col}' not found in the test DataFrame."
                )
            required_cols.append(strata_col)

        # Filter relevant columns
        X = X_input[required_cols]

        model = self.get_model()
        # Generate predictions based on model type
        if isinstance(model, (CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter)):

            if isinstance(model, CoxPHFitter):
                predictions = -self.model.predict_partial_hazard(X)
            else:
                predictions = self.model.predict_median(X)

            # Compute concordance index
            c_index = concordance_index(
                X[duration_col], predictions, X[event_col]
            ).item()

        elif isinstance(model, KaplanMeierFitter):
            prediction = [model.median_survival_time_] * len(X)
            c_index = concordance_index(
                X[duration_col], prediction, X[event_col]
            ).item()

        elif isinstance(model, GradientBoostingSurvivalAnalysis):
            X_input_ = X_input[self.features]
            model: GradientBoostingSurvivalAnalysis = self.model
            predictions = model.predict(X_input_)
            c_index = concordance_index_censored(
                X_input[event_col].astype(bool), X_input[duration_col], predictions
            )[0]

        else:
            raise NotImplementedError(
                f"Model type {self.model_type} not supported for evaluation."
            )

        if save_metric:
            if evaluation_set == "test":
                self.key_test_performance_metrics["c_index"] = c_index
            elif evaluation_set == "train":
                self.key_train_performance_metrics["c_index"] = c_index
            elif evaluation_set == "cv":
                self.key_validation_performance_metrics["c_index"] = c_index

        return c_index

    def evaluate_brier_score(
        self,
        X_train,
        X_test,
        evaluation_times=None,
        save_metric=False,
        evaluation_set="test",
    ):
        duration_col = self.duration_col
        event_col = self.event_col

        if duration_col not in X_train.columns or event_col not in X_train.columns:
            raise ValueError(
                f"Columns '{duration_col}' and '{event_col}' must be present in the training DataFrame."
            )

        if evaluation_times is None:
            # Default evaluation times if not provided
            evaluation_times = np.array([30, 60, 90, 365, 730, 1000])

        model = self.get_model()

        if not isinstance(
            model,
            (
                CoxPHFitter,
                WeibullAFTFitter,
                GradientBoostingSurvivalAnalysis,
                LogNormalAFTFitter,
                KaplanMeierFitter,
            ),
        ):
            raise ValueError(
                "Brier score evaluation is only supported for CoxPHFitter,"
                " WeibullAFTFitter, KaplanMeierFitter and GradientBoostingSurvivalAnalysis models."
            )

        # Step 1: Calculate survival functions for the test set
        surv_funcs = self.predict_survival(X_test, times=evaluation_times)

        # Step 2: Prepare survival datasets
        y_train = Surv.from_dataframe(event_col, duration_col, X_train)
        y_test = Surv.from_dataframe(event_col, duration_col, X_test)

        if isinstance(model, KaplanMeierFitter):
            # For KaplanMeierFitter, there is only one survival function.
            # The survival function is the same for all patients in the test set.
            surv_prob_list = [surv_funcs.iloc[:, 0].values] * len(X_test)
        else:
            # Convert columns (patients) into list of arrays (each array = survival probs over evaluation times)
            surv_prob_list = [surv_funcs[col].values for col in surv_funcs.columns]

        # Step 4: Compute Brier scores at the specified times
        times, brier_scores = brier_score(
            y_train, y_test, surv_prob_list, evaluation_times
        )

        # Step 5: Package results into DataFrame
        brier_df = pd.DataFrame({"time": times, "brier_score": brier_scores})

        if save_metric:
            if evaluation_set == "test":
                self.key_test_performance_metrics["avg_brier_score"] = np.mean(
                    brier_scores
                )
            elif evaluation_set == "train":
                self.key_train_performance_metrics["avg_brier_score"] = np.mean(
                    brier_scores
                )

        return brier_df

    def evaluate_cumulative_dynamic_auc(
        self,
        X_train,
        X_test,
        evaluation_times,
        save_metric=False,
        evaluation_set="test",
    ):
        """
        Evaluate the Area Under the Curve (AUC) for survival predictions over specified evaluation times.
        This method calculates the cumulative dynamic AUC for survival models, which measures the
        discriminative ability of the model at different time points. It supports models such as
        CoxPHFitter, WeibullAFTFitter, GradientBoostingSurvivalAnalysis, and KaplanMeierFitter.
        Parameters:
        ----------
        X_train : pd.DataFrame
            Training dataset containing features used for model training.
        X_test : pd.DataFrame
            Test dataset containing features used for evaluation.
        evaluation_times : array-like or None
            Array of time points at which to evaluate the AUC. If None, default evaluation times
            [30, 60, 90, 365, 730, 1000] are used.
        Returns:
        -------
        auc_df : pd.DataFrame
            A DataFrame containing the evaluation times and their corresponding AUC values.
            Columns: ["time", "auc"].
        mean_auc : float
            The mean AUC value across all evaluation times.
        Raises:
        ------
        ValueError
            If the model is not one of the supported types or if there are missing evaluation times
            in the survival functions.
        Notes:
        ------
        - The method uses `cumulative_dynamic_auc` from `scikit-survival` to compute the AUC.
        - For KaplanMeierFitter, the survival function is assumed to be the same for all patients
          in the test set.
        """

        duration_col = self.duration_col
        event_col = self.event_col

        if duration_col not in X_train.columns or event_col not in X_train.columns:
            raise ValueError(
                f"Columns '{duration_col}' and '{event_col}' must be present in the training DataFrame."
            )

        if evaluation_times is None:
            # Default evaluation times if not provided
            evaluation_times = np.array([30, 60, 90, 365, 730, 1000])

        model = self.get_model()
        if not isinstance(
            model,
            (
                CoxPHFitter,
                WeibullAFTFitter,
                GradientBoostingSurvivalAnalysis,
                KaplanMeierFitter,
                LogNormalAFTFitter,
            ),
        ):
            raise ValueError(
                "Brier score evaluation is only supported for CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, KaplanMeierFitter and GradientBoostingSurvivalAnalysis models."
            )

        # Step 1: Calculate survival functions for the test set
        surv_funcs = self.predict_survival(X_test, times=evaluation_times)
        missing = set(evaluation_times) - set(surv_funcs.index)
        if missing:
            raise ValueError(f"Missing times in surv_funcs: {missing}")

        # Step 2: Prepare survival datasets
        y_train = Surv.from_dataframe(event_col, duration_col, X_train)
        y_test = Surv.from_dataframe(event_col, duration_col, X_test)

        if isinstance(model, KaplanMeierFitter):
            # For KaplanMeierFitter, there is only one survival function.
            # The survival function is the same for all patients in the test set.
            risk_score_list = [1 - surv_funcs.iloc[:, 0].values] * len(X_test)
        else:
            # Convert columns (patients) into list of arrays (each array = survival probs over evaluation times)
            risk_score_list = [1 - surv_funcs[col].values for col in surv_funcs.columns]

        auc_over_time, mean_auc = cumulative_dynamic_auc(
            y_train, y_test, risk_score_list, times=evaluation_times
        )

        # Step 4: Collect results
        auc_df = pd.DataFrame({"time": evaluation_times, "auc": auc_over_time})

        if save_metric:
            if evaluation_set == "test":
                self.key_test_performance_metrics["avg_cumulative_dynamic_auc"] = (
                    mean_auc
                )
            elif evaluation_set == "train":
                self.key_train_performance_metrics["avg_cumulative_dynamic_auc"] = (
                    mean_auc
                )

        return auc_df, mean_auc
    
    def pred_survival_prob_true_duration(
        self, X_input: pd.DataFrame, conditional_after=None
    ):
        """
        Get survival probabilities for the test set.

        Args:
            X_input: DataFrame containing test features.
            true_events_durations: Optional DataFrame containing true event durations and events.
            times: Optional array of times at which to compute survival probabilities.
            conditional_after: Optional time after which to condition the survival function.

        Returns:
            A DataFrame with survival probabilities.
        """
        true_duration_col = self.duration_col
        true_event_col = self.event_col

        if (
            true_event_col not in X_input.columns
            or true_duration_col not in X_input.columns
        ):
            raise ValueError(
                f"Columns '{true_duration_col}' and '{true_event_col}' must be present in the training DataFrame."
            )

        # P(event = 0 until T) = S(T)
        # P(event = 1 until T) = 1 - S(T)
        # If event happens at T, P(event = 1 until T + 1) = 1 -> S(T+1) = 0

        surv_prob_list = []
        event_id_col = self.event_id_col
        surv_funcs = self.predict_survival(X_input, conditional_after=conditional_after)

        if event_id_col in X_input.columns:
            X_input = X_input.set_index(event_id_col)

        if isinstance(self.model, KaplanMeierFitter):
            true_duration_plus_one = X_input[true_duration_col]
            surv_func = surv_funcs.iloc[
                :, 0
            ]  # For KaplanMeierFitter, we have a single survival function
            closest_time = surv_func.index.get_indexer(
                true_duration_plus_one, method="nearest"
            )
            closest_time = surv_func.index[closest_time]
            surv_prob = surv_func.loc[closest_time].values
            surv_prob_list = surv_prob.tolist()
        else:
            for idx, subject_index_col in enumerate(surv_funcs.columns):
                surv_func = surv_funcs[subject_index_col]
                true_duration_plus_one = X_input.loc[subject_index_col][
                    true_duration_col
                ]
                closest_time = surv_func.index.get_indexer(
                    [true_duration_plus_one], method="nearest"
                )
                closest_time = surv_func.index[closest_time[0]]
                surv_prob = surv_func.loc[closest_time].item()
                surv_prob_list.append(surv_prob)

        result_df = pd.DataFrame(
            {
                f"{self.event_id_col}": X_input.index.values,
                "true_duration": X_input[true_duration_col].values,
                "true_event": X_input[true_event_col].values,
                "survival_prob_at_true_duration": surv_prob_list,
                "prob_event": 1 - np.array(surv_prob_list),
            }
        )

        return result_df

    def evaluate_model_at_true_duration(
        self, X_test: pd.DataFrame, conditional_after=None, plot_roc_curve=True
    ):
        X_test = X_test.copy()
        X_test[self.duration_col] = (
            X_test[self.duration_col] + 1
        )  # Increment duration by 1 to get S(T+1)
        surv_around_true_duration_df = self.pred_survival_prob_true_duration(
            X_test, conditional_after=conditional_after
        )

        surv_around_true_duration_df = surv_around_true_duration_df.rename(
            columns={
                "survival_prob_at_true_duration": "survival_prob_at_true_duration_plus_one",
                "true_duration": "true_duration_plus_one",
            }
        )

        surv_around_true_duration_df[
            "prob_event_happened_before_true_duration_plus_one"
        ] = (
            1 - surv_around_true_duration_df["survival_prob_at_true_duration_plus_one"]
        )

        # Calculate AUC ROC
        # If event happened at T, P_true(event = 1 until T+1) = 1, else = 0
        labels = surv_around_true_duration_df[
            "true_event"
        ]  # Invert Labels. S_true(T+1) = 0 if event = 1 (happened) at T, else = 1
        scores = surv_around_true_duration_df[
            "prob_event_happened_before_true_duration_plus_one"
        ]  # S(T+1) gives P(event = 0 until T+1)
        auc_roc = roc_auc_score(labels, scores)

        if plot_roc_curve:
            plot_auc(
                scores,
                labels,
                title="AUC ROC for Event Probability around True Event Durations",
            )

        return surv_around_true_duration_df, auc_roc

    def evaluate_model_at_time_t(
        self,
        X_input: pd.DataFrame,
        t: float,
        conditional_after=None,
        save_metric=False,
        evaluation_set="test",
    ):
        """
        Evaluate the model at a specific time t by calculating the AUC ROC. If a event was censored before t,
        the event is not considered in the evaluation.

        Args:
            X_input: DataFrame containing test features.
            t: Time at which to evaluate the model.
            conditional_after: Optional time after which to condition the survival function.
            plot_roc_curve: Whether to plot the ROC curve (default is True).

        Returns:
            AUC ROC score at time t.
        """

        surv_around_t_df = self.predict_events_at_t(
            X_input, t=t, conditional_after=conditional_after
        )
        surv_around_t_df = surv_around_t_df[surv_around_t_df["event_at_t"].notna()]
        surv_around_t_df["event_at_t"] = surv_around_t_df["event_at_t"].astype(int)

        # Calculate AUC ROC
        labels = 1 - surv_around_t_df["event_at_t"]
        scores = surv_around_t_df["survival_prob_at_t"]
        auc_roc = roc_auc_score(labels, scores)

        if save_metric:
            metric_name = f"auc_roc_at_t_{t}"
            if evaluation_set == "test":
                self.key_test_performance_metrics[metric_name] = auc_roc
            elif evaluation_set == "train":
                self.key_train_performance_metrics[metric_name] = auc_roc

        return surv_around_t_df, auc_roc

    def plot_survival_function(
        self,
        X_test,
        times=None,
        conditional_after=None,
        n=5,
        duration_col=None,
        event_col=None,
        show_plot: bool = True,
        title: str = "Survival Functions"
    ):
        true_events_durations_df = None
        event_id_col = self.event_id_col
        surv_funcs = self.predict_survival(
            X_test, times=times, conditional_after=conditional_after
        )
        if event_id_col in X_test.columns:
            X_test = X_test.set_index(event_id_col)
        if self.model_type != SurvivalModelType.KAPLAN_MEIER:
            ids = surv_funcs.columns[:n]
            X_test = X_test.loc[ids]
        else:
            X_test = X_test.iloc[:n]
        if duration_col is not None and event_col is not None:
            true_events_durations_df = self.pred_survival_prob_true_duration(
                X_test, conditional_after=conditional_after
            )
            return plot_survival_function_plotly(
                surv_funcs,
                n=n,
                true_events_durations_df=true_events_durations_df,
                event_id_col=event_id_col,
                show_plot=show_plot,
                title=title,
            )

    def get_model(
        self,
    ) -> (
        KaplanMeierFitter
        | CoxPHFitter
        | WeibullAFTFitter
        | LogNormalAFTFitter
        | GradientBoostingSurvivalAnalysis
    ):
        if self.model is None:
            raise ValueError(
                "Model has not been trained yet. Call fit() before using the model."
            )

        return self.model

    def save_key_performance_metrics(self):
        """
        Save key performance metrics to a txt file.
        """
        key_test_metrics = self.key_test_performance_metrics
        key_train_metrics = self.key_train_performance_metrics

        model_name = (
            self.model_name if self.model_name else "next_event_prediction_model"
        )
        model_path = self.model_path if self.model_path else "."

        model_name = model_name.replace(" ", "_").replace("/", "_").lower()

        path = f"{model_path}/{model_name}"

        os.makedirs(path, exist_ok=True)

        file_path = os.path.join(path, "key_performance_metrics.txt")

        with open(file_path, "w") as f:
            f.write("Key Test Performance Metrics:\n")
            for metric, value in key_test_metrics.items():
                f.write(f"{metric}: {value}\n")

            f.write("\nKey Train Performance Metrics:\n")
            for metric, value in key_train_metrics.items():
                f.write(f"{metric}: {value}\n")

    def save_random_search_cv_results(self):
        """
        Save random search cross-validation results to a txt file.
        """
        save_dir = self.get_model_dir()

        if self.random_search_cv_results is None:
            raise ValueError(
                "Random search CV results have not been computed yet. Please run random search first."
            )

        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, "random_search_cv_results.txt")

        with open(file_path, "w") as f:
            f.write("Random Search CV Results:\n")
            for metric, value in self.random_search_cv_results.items():
                f.write(f"{metric}: {value}\n")

    def get_model_dir(self):
        """
        Get the model directory where the model .pkl file will be saved.
        """
        if self.model_path is None:
            return "."

        model_name = (
            self.model_name if self.model_name else "next_event_prediction_model"
        )
        model_name = model_name.replace(" ", "_").replace("/", "_").lower()

        return os.path.join(self.model_path, model_name)

    def save_model(self):
        if self.model is None:
            raise ValueError(
                "Model has not been trained yet. Please call train() first."
            )

        model_dir = self.get_model_dir()

        os.makedirs(model_dir, exist_ok=True)

        model_name = (
            self.model_name if self.model_name else "next_event_prediction_model"
        )
        model_name = model_name.replace(" ", "_").replace("/", "_").lower()

        file_path = os.path.join(model_dir, f"{model_name}.pkl")

        print(f"Saving model to {file_path}")

        with open(file_path, "wb") as f:
            pickle.dump(self, f)

            return file_path

    def save_model_params(self):
        """
        Save model parameters to a txt file.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call fit() first.")

        model_name = (
            self.model_name if self.model_name else "next_event_prediction_model"
        )
        model_path = self.model_path if self.model_path else "."

        model_name = model_name.replace(" ", "_").replace("/", "_").lower()

        path = f"{model_path}/{model_name}"

        os.makedirs(path, exist_ok=True)

        file_path = os.path.join(path, "model_params.txt")

        with open(file_path, "w") as f:
            f.write("Features:\n")
            for feature in self.features:
                f.write(f"{feature}\n")
            model_type_str = (
                self.model_type.name if self.model_type else "Not specified"
            )
            f.write(f"\nModel Type: {model_type_str}\n")
            f.write("Additional Model Parameters:\n")
            for param, value in self.model_params.items():
                f.write(f"{param}: {value}\n")
            f.write(
                "\nStrata Column: {}\n".format(
                    self.strata_col if self.strata_col else "Not specified"
                )
            )
            f.write(
                "Cluster Column: {}\n".format(
                    self.cluster_col if self.cluster_col else "Not specified"
                )
            )
            if self.model_type in [
                SurvivalModelType.COX_PH,
                SurvivalModelType.WEIBULL_AFT,
                SurvivalModelType.LOGNORMAL_AFT,
            ]:
                f.write(
                    "Number of Observations: {}\n".format(
                        self.model._n_examples
                        if hasattr(self.model, "_n_examples")
                        else "Not available"
                    )
                )

        if self.model_type in [
            SurvivalModelType.COX_PH,
            SurvivalModelType.WEIBULL_AFT,
            SurvivalModelType.LOGNORMAL_AFT,
        ]:
            file_path = os.path.join(path, "model_summary.csv")
            self.model.summary.to_csv(file_path)


class NextEventSurvivalWrapper(BaseEstimator):
    def __init__(self, model_config, columns_order: list, model_params=None):
        self.model_config = model_config
        self.columns_order = columns_order
        self.model_params = model_params if model_params is not None else {}
        self.model = None

    def fit(self, X, y=None):
        model_config = self.model_config
        model_config["model_params"] = self.model_params

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_order)

        self.model = NextEventPredictionModel(model_config)
        self.model.fit(X)

        return self

    def score(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_order)
        return self.model.evaluate_c_index(X, evaluation_set="cv")
