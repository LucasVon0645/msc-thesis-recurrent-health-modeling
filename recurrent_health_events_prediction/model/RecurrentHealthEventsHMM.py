import os
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm
import torch
import pickle

from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal, Poisson, Bernoulli, Categorical, Gamma, HalfNormal, Exponential, StudentT, IndependentComponents

from recurrent_health_events_prediction.model.utils import get_expected_value, normalize_probabilities, predict_most_likely_state_single_obs, predict_partial_obs_given_history_proba, predict_single_partial_obs_proba, predict_single_obs_proba
from recurrent_health_events_prediction.model.model_types import DistributionType
import yaml

class RecurrentHealthEventsHMM:
    def __init__(self, config: dict, n_states: Optional[int] = None, features: Optional[dict] = None, add_distribution_params: Optional[dict] = None, id_col: str = 'SUBJECT_ID', time_col: str = 'ADMITTIME'):
        self.n_states: int = n_states or config.get("n_states")
        if not self.n_states:
            raise ValueError("Number of states (n_states) must be provided either as an argument or in the config.")

        self.features = features or config.get("features")
        if not self.features:
            raise ValueError("Features must be provided either as an argument or in the config.")

        self.add_distribution_params = add_distribution_params or config.get("add_distribution_params")
        self.id_col: str = config.get("id_col", id_col)
        self.time_col: str = config.get("time_col", time_col)

        self.config: dict = config
        self.dof: Optional[int] = config.get("dof")  # Degrees of freedom for Student's t-distribution

        self.model = None
        self.feature_dist = {feature: [None] * self.n_states for feature in self.features.keys()}
        self.hidden_state_labels = None  # To store labels for hidden states
        self.apply_power_transform: bool = config.get("apply_power_transform", False)
        self.power_transform_scalers: dict[str, PowerTransformer] = {} # Placeholder for power transform scalers if needed
        self.power_transform_columns: list[str] = self.config.get("power_transform_variables", [])

        if self.apply_power_transform:
            if not self.power_transform_columns:
                raise ValueError(
                    "Power transform columns must be specified in the config if apply_power_transform is True."
                )

    def _create_state_distributions(self):
        states_distributions = []
        for state_idx in range(self.n_states):
            distributions = []
            for feature, dist_type in self.features.items():
                try:
                    dist_type = DistributionType(dist_type)
                except ValueError:
                    raise ValueError(f"Unsupported distribution type: {dist_type}. Supported types are: {list(DistributionType)}")

                if dist_type in [DistributionType.GAUSSIAN, DistributionType.NORMAL, DistributionType.LOG_NORMAL]:
                    normal_dist = Normal()
                    self.feature_dist[feature][state_idx] = normal_dist
                    distributions.append(normal_dist)
                elif dist_type == DistributionType.POISSON:
                    poisson_dist = Poisson()
                    self.feature_dist[feature][state_idx] = poisson_dist
                    distributions.append(poisson_dist)
                elif dist_type == DistributionType.BERNOULLI:
                    bernoulli_dist = Bernoulli()
                    self.feature_dist[feature][state_idx] = bernoulli_dist
                    distributions.append(bernoulli_dist)
                elif dist_type == DistributionType.CATEGORICAL:
                    add_distribution_params = self.add_distribution_params
                    n_categories = None
                    if add_distribution_params is not None:
                        if feature in add_distribution_params:
                            n_categories = self.add_distribution_params[feature].get('n_categories')
                        probs_init = np.random.dirichlet(np.ones(n_categories)) if n_categories is not None else None
                        probs_init = probs_init.reshape(-1, n_categories) if probs_init is not None else None
                        categorical_dist = Categorical(n_categories=n_categories, probs=probs_init)
                    else:
                        categorical_dist = Categorical()
                    self.feature_dist[feature][state_idx] = categorical_dist
                    distributions.append(categorical_dist)
                elif dist_type == DistributionType.GAMMA:
                    gamma_dist = Gamma()
                    self.feature_dist[feature][state_idx] = gamma_dist
                    distributions.append(gamma_dist)
                elif dist_type == DistributionType.STUDENT_T:
                    if self.dof is None:
                        raise ValueError("Degrees of freedom (dof) must be provided for Student's t-distribution.")
                    student_t_dist = StudentT(dofs=self.dof)
                    self.feature_dist[feature][state_idx] = student_t_dist
                    distributions.append(student_t_dist)
                else:
                    raise ValueError(f"Unsupported distribution type: {dist_type}")
            if len(distributions) == 1:
                # If only one distribution, no need for IndependentComponents
                state_dist = distributions[0]
            else:
                state_dist = IndependentComponents(distributions)
            states_distributions.append(state_dist)
        return states_distributions

    def fit(
        self,
        sequences: list[tuple],
        random_state=42,
        verbose=False,
        initialize_from_first_obs_with_gmm: bool = False,
    ):
        if initialize_from_first_obs_with_gmm:
            self.model = DenseHMM(
                distributions=self.initial_emission_distributions,
                starts=self.initial_probs,
                edges=self.initial_transitions,
                random_state=random_state,
                verbose=verbose,
            )
        else:
            self.model = DenseHMM(random_state=random_state, verbose=verbose)
            states_distributions = self._create_state_distributions()
            self.model.add_distributions(states_distributions)
        if verbose:
            print(f"Fitting Hidden Markov Model with {len(sequences)} sequences...")
        self.model.fit(sequences)

    def _categorical_dist_present(self):
        """
        Returns True if any feature uses a categorical or binomial distribution.
        """
        return any(
            DistributionType(dist_type) in [DistributionType.CATEGORICAL]
            for dist_type in self.features.values()
        )

    def transform_dataframe(self, df: pd.DataFrame):
        sequences = []
        id_col = self.id_col
        time_col = self.time_col
        grouped = df.sort_values(by=[id_col, time_col]).groupby(id_col)

        categorical_dist_present = self._categorical_dist_present()

        for _, group in grouped:
            sequence = []
            for _, row in group.iterrows():
                obs = []
                for feature, dist_type in self.features.items():
                    try:
                        dist_type = DistributionType(dist_type)
                    except ValueError:
                        raise ValueError(f"Unsupported distribution type: {dist_type}. Supported types are: {list(DistributionType)}")
                    value = row[feature]
                    if categorical_dist_present:
                        # pomegranate doesnt support mixed types in sequences, so we convert all values to int
                        # categorical distributions only accept integers
                        obs.append(int(round(value)) if not pd.isna(value) else None)
                    else:
                        obs.append(float(value))
                sequence.append(obs)
            sequences.append(sequence)

        return sequences  # Important: do NOT wrap in np.array()!

    def fit_transform_power_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform specified columns independently using PowerTransformer (Yeo-Johnson).
        Stores one transformer per column. Leaves NaNs untouched.
        """
        df = df.copy()
        power_vars = self.power_transform_columns
        applicable_vars = [col for col in power_vars if col in df.columns]

        if not applicable_vars:
            return df

        self.power_transform_scalers = {}  # Dict to store one transformer per column

        for col in applicable_vars:
            col_data = df[col]
            mask = col_data.notna()

            pt = PowerTransformer(method='yeo-johnson')
            transformed = pt.fit_transform(col_data[mask].values.reshape(-1, 1)).ravel()

            # Save the fitted transformer
            self.power_transform_scalers[col] = pt

            # Replace only non-NaN values in the DataFrame
            df.loc[mask, col] = transformed

        # Save which columns were transformed
        self.power_transform_columns = applicable_vars

        return df

    def transform_with_fitted_power(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using previously fitted PowerTransformers (one per column).
        Transforms only non-NaN values. Leaves NaNs untouched.
        """
        df = df.copy()

        if not self.power_transform_scalers or not self.power_transform_columns:
            raise RuntimeError("PowerTransformers have not been fitted. Call fit_transform_power_variables() first.")

        for col in self.power_transform_columns:
            if col not in df.columns or col not in self.power_transform_scalers:
                continue

            col_data = df[col]
            mask = col_data.notna()

            pt = self.power_transform_scalers[col]
            transformed = pt.transform(col_data[mask].values.reshape(-1, 1)).ravel()

            df.loc[mask, col] = transformed

        return df

    def inverse_transform_power(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the columns using the fitted PowerTransformer.
        Only applies to columns that were fitted and are still present in df.
        Overwrites transformed columns in-place with their original scale.
        """
        if self.power_transform_scaler is None or not self.power_transform_columns:
            raise RuntimeError("PowerTransformer has not been fitted. Call fit_transform_power_variables() first.")

        applicable_vars = [col for col in self.power_transform_columns if col in df.columns]
        if not applicable_vars:
            return df

        original = self.power_transform_scaler.inverse_transform(df[applicable_vars])
        original_df = pd.DataFrame(original, columns=applicable_vars, index=df.index)

        df.update(original_df)
        return df

    def initialize_with_first_obs_with_gmm(self, first_obs_df, verbose=False, random_state=42):
        if verbose:
            print("Initializing model with first observations using GMM...")
        distributions = self._create_state_distributions()
        feature_names = list(self.features.keys())
        first_obs_data = first_obs_df[feature_names].values

        categorical_dist_present = self._categorical_dist_present()
        if categorical_dist_present:
            first_obs_data = first_obs_data.astype(int)  # Convert to int for categorical distributions

        gmm_clustering = GeneralMixtureModel(distributions, verbose=verbose, random_state=random_state)
        gmm_clustering.fit(first_obs_data)

        self.initial_probs = gmm_clustering.priors
        self.initial_emission_distributions = gmm_clustering.distributions

        # Initialize transition matrix randomly
        rng = np.random.default_rng(random_state)
        n_states = self.n_states
        transitions = rng.uniform(size=(n_states, n_states))
        transitions /= transitions.sum(axis=1, keepdims=True)
        self.initial_transitions = transitions.tolist()

    def train(self, df, verbose=False, random_state=42, initialize_from_first_obs_with_gmm=False):
        if initialize_from_first_obs_with_gmm:
            first_obs_df = df.groupby(self.id_col).first().reset_index()
            self.initialize_with_first_obs_with_gmm(first_obs_df, verbose, random_state=random_state)

            sequences_gt_length_one_df = df.groupby(self.id_col).filter(lambda x: len(x) > 1)
            sequences = self.transform_dataframe(sequences_gt_length_one_df)
        else:
            sequences = self.transform_dataframe(df)

        self.fit(sequences, random_state=random_state, verbose=verbose, initialize_from_first_obs_with_gmm=initialize_from_first_obs_with_gmm)

    def infer_hidden_states(self, df):
        if self.model is None or not self.model.distributions:
            raise ValueError("Model has not been trained yet. Please call train() first.")
        sequences = self.transform_dataframe(df)
        hidden_states_seq = []
        for sequence in sequences:
            if len(sequence) == 1:
                obs = sequence[0]
                hidden_states = predict_most_likely_state_single_obs(self.model, obs)
                hidden_states_seq.append([hidden_states])  # shape: [1 x num_states]

            else:
                hidden_states = self.model.predict([sequence]) # returns pytorch.tensor
                hidden_states_seq.append(hidden_states.to('cpu').tolist()[0])  # Convert to list for consistency

        return hidden_states_seq

    def predict_proba(self, df):
        sequences = self.transform_dataframe(df)
        hidden_states_seq_list = []
        for sequence in sequences:
            if len(sequence) == 1:
                obs = sequence[0]
                joint_probs = predict_single_obs_proba(self.model, obs)
                hidden_states_seq_list.append([joint_probs])  # shape: [1 x num_states]
            else:
                hidden_states = self.model.predict_proba([sequence]) # returns pytorch.tensor
                hidden_states_seq_list.append(hidden_states.to('cpu').tolist()[0])  # Convert to list for consistency
        return hidden_states_seq_list

    def predict_proba_last_obs_partial(self, df, data_already_transformed: bool = False):
        sequences = self.transform_dataframe(df)
        hidden_states_seq_list = []
        for sequence in sequences:
            if len(sequence) == 1:
                obs = sequence[0]
                state_probs = predict_single_partial_obs_proba(self.model, obs)
                hidden_states_seq_list.append(state_probs)  # shape: [1 x num_states]
            else:
                hidden_states = predict_partial_obs_given_history_proba(self.model, sequence[0:-1], sequence[-1])
                hidden_states_seq_list.append(hidden_states)
        return hidden_states_seq_list

    def log_likelihood(self, df, total: bool = True) -> tuple[float, int] | tuple[list[float], int]:
        """
        Calculate the log-likelihood of the model given the data in df.
        If total is True, returns the total log-likelihood and number of samples.
        If total is False, returns a list of log-likelihoods for each sequence and the number of samples.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call train() first.")
        sequences = self.transform_dataframe(df)
        log_likelihoods = []
        for sequence in sequences:
            log_likelihood = self.model.log_probability([sequence])
            log_likelihoods.append(log_likelihood.item())

        num_samples = len(log_likelihoods)

        if total:
            return np.sum(log_likelihoods).item(), num_samples
        else:
            return log_likelihoods, num_samples

    def get_model_params_str(self):
        if self.model is None:
            return "Model has not been trained yet."

        hidden_state_labels = self.hidden_state_labels

        lines = []
        feature_names = list(self.features.keys())
        lines.append("\nModel Parameters:")
        lines.append("=" * 50)
        lines.append(f"Number of States: {self.n_states}")
        lines.append(("Hidden State Labels: ") if hidden_state_labels is not None else "Hidden State Labels: Not defined")
        if hidden_state_labels is not None:
            for state_idx, label in hidden_state_labels.items():
                lines.append(f"  {int(state_idx)}: {label}")
        lines.append(f"Feature Names: {feature_names}")
        lines.append("Transition Matrix:")
        lines.append("  " + str(self.get_transition_matrix()))
        lines.append("Initial Probabilities:")
        lines.append("  " + str(self.get_initial_probabilities()))
        lines.append("End Probabilities:")
        lines.append("  " + str(self.get_end_probabilities()))
        lines.append("Distributions:")
        for state_idx, state_dist in enumerate(self.model.distributions):
            lines.append("-" * 50)
            if hidden_state_labels is not None:
                state_label = hidden_state_labels.get(state_idx, f"State {state_idx}")
                lines.append(f"State {state_idx} ({state_label}):")
            else:
                lines.append(f"State {state_idx}:")
            if not isinstance(state_dist, IndependentComponents):
                if isinstance(state_dist, Normal):
                    lines.append(f"  Mean: {state_dist.means.numpy().tolist()}")
                    lines.append(f"  Covariance: {state_dist.covs.numpy()}")
                elif isinstance(state_dist, StudentT):
                    lines.append(f"  Degrees of Freedom: {state_dist.dofs.numpy().tolist()}")
                    lines.append(f"  Mean: {state_dist.means.numpy().tolist()}")
                    lines.append(f"  Covariance: {state_dist.covs.numpy().tolist()}")
                elif isinstance(state_dist, Poisson):
                    lines.append(f"  Lambda: {state_dist.lambdas.numpy().tolist()}")
                elif isinstance(state_dist, Categorical):
                    add_distribution_params = self.add_distribution_params.get(feature_names[0])
                    labels = add_distribution_params.get('labels') if add_distribution_params else None
                    if labels:
                        lines.append(f"  Labels: {add_distribution_params['labels']}")
                    lines.append(f"  Probabilities: {state_dist.probs.numpy()[0].tolist()}")
                    lines.append(f"  Expected value: {round(get_expected_value(state_dist), 2)}")
                elif isinstance(state_dist, Gamma):
                    lines.append(f"  Shape: {state_dist.shapes.numpy().tolist()}")
                    lines.append(f"  Rate: {state_dist.rates.numpy().tolist()}")
                elif isinstance(state_dist, HalfNormal):
                    lines.append(f"  Covariance: {state_dist.covs.numpy().tolist()}")
                elif isinstance(state_dist, Exponential):
                    lines.append(f"  Lambda: {state_dist.scales.numpy().tolist()}")
            else:
                for feature_idx, dist in enumerate(state_dist.distributions):
                    lines.append(f"  Feature: {feature_names[feature_idx]} ({dist.name})")
                    try:
                        dist_type = DistributionType(self.features[feature_names[feature_idx]])
                    except ValueError:
                        raise ValueError(f"Unsupported distribution type: {self.features[feature_names[feature_idx]]}. Supported types are: {list(DistributionType)}")
                    if isinstance(dist, Normal):
                        lines.append(f"    Mean: {dist.means.numpy().tolist()}")
                        lines.append(f"    Covariance:")
                        lines.append(f"      {dist.covs.numpy()}")
                    elif isinstance(dist, StudentT):
                        lines.append(f"    Degrees of Freedom: {dist.dofs.numpy().tolist()}")
                        lines.append(f"    Mean: {dist.means.numpy().tolist()}")
                        lines.append(f"    Covariance: {dist.covs.numpy().tolist()}")
                    elif isinstance(dist, Poisson):
                        lines.append(f"    Lambda: {dist.lambdas.numpy().tolist()}")
                    elif isinstance(dist, Bernoulli):
                        lines.append(f"    Probability: {dist.probs.numpy().tolist()}")
                    elif isinstance(dist, Gamma):
                        lines.append(f"    Shape: {dist.shapes.numpy().tolist()}")
                        lines.append(f"    Rate: {dist.rates.numpy().tolist()}")
                    elif isinstance(dist, HalfNormal):
                        lines.append(f"    Scale: {dist.scales.numpy().tolist()}")
                    elif isinstance(dist, Exponential):
                        lines.append(f"    Lambda: {dist.lambdas.numpy().tolist()}")
                    elif isinstance(dist, Categorical):
                        add_distribution_params = self.add_distribution_params.get(feature_names[feature_idx])
                        labels = add_distribution_params.get('labels') if add_distribution_params else None
                        if labels:
                            lines.append(f"    Labels: {add_distribution_params['labels']}")
                        lines.append(f"    Probabilities: {dist.probs.numpy()[0].tolist()}")
                        lines.append(f"    Expected value: {round(get_expected_value(dist), 2)}")
        return '\n'.join(lines)

    def print_model_params(self):
        """
        Print the model parameters in a human-readable format.
        """
        params_str = self.get_model_params_str()
        print(params_str)

    def get_features_dist_df(self, include_state_labels: bool = False):
        """
        Returns a DataFrame with the distribution parameters for numerical features (Gaussian, Poisson).
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call train() first.")

        columns = [('State', '')]
        data = []

        if include_state_labels:
            if self.hidden_state_labels is None:
                raise ValueError("Hidden state labels have not been defined. Please call define_hidden_state_labels() first.")
            columns.append(('State Label', ''))

        # Dynamically create column headers based on feature names and distribution types
        for feature_name, dist_type in self.features.items():
            try:
                dist_type = DistributionType(dist_type)
            except ValueError:
                raise ValueError(f"Unsupported distribution type: {dist_type}. Supported types are: {list(DistributionType)}")
            if dist_type in [DistributionType.GAUSSIAN, DistributionType.NORMAL, DistributionType.LOG_NORMAL]:
                columns.extend([(feature_name, 'Mean'), (feature_name, 'Cov')])
            elif dist_type == DistributionType.STUDENT_T:
                columns.append((feature_name, 'Dof'))
                columns.append((feature_name, 'Mean'))
                columns.append((feature_name, 'Cov'))
            elif dist_type == DistributionType.POISSON:
                columns.append((feature_name, 'Lambda'))
            elif dist_type == DistributionType.CATEGORICAL:
                # For categorical distributions, we need to handle labels and probabilities
                # Check if add_distribution_params is provided for this feature
                add_distribution_params = self.add_distribution_params.get(feature_name)
                labels = add_distribution_params.get('labels') if add_distribution_params else None
                if labels:
                    col_name = f"Probabilities [{', '.join(labels)}]"
                    columns.append((feature_name, col_name))
                else:
                    columns.append((feature_name, 'Probabilities'))
                columns.append((feature_name, 'Expected Value'))
            elif dist_type == DistributionType.BERNOULLI:
                columns.append((feature_name, 'Probability'))
            elif dist_type == DistributionType.GAMMA:
                columns.append((feature_name, 'Shape'))
                columns.append((feature_name, 'Rate'))
                columns.append((feature_name, 'Expected Value'))
            elif dist_type == DistributionType.HALF_NORMAL:
                columns.append((feature_name, 'Cov'))
            else:
                raise ValueError(f"Unsupported distribution type: {type(dist_type)}. Supported types are: {list(DistributionType)}")

        # Populate the data
        for state_idx, state_dist in enumerate(self.model.distributions):
            if include_state_labels:
                state_label = self.hidden_state_labels[state_idx]
                row = [state_idx, state_label]
            else:
                row = [state_idx]

            if not isinstance(state_dist, IndependentComponents):
                if isinstance(state_dist, Normal):
                    row.extend([state_dist.means.numpy().item(), state_dist.covs.numpy().item()])
                elif isinstance(state_dist, Poisson):
                    row.append(state_dist.lambdas.numpy().item())
                elif isinstance(state_dist, Categorical):
                    row.append(state_dist.probs.numpy()[0].tolist())
                    row.append(round(get_expected_value(state_dist), 2))
                elif isinstance(state_dist, Bernoulli):
                    row.append(state_dist.probs.numpy().item())
                elif isinstance(state_dist, Gamma):
                    row.append(state_dist.shapes.numpy().item())
                    row.append(state_dist.rates.numpy().item())
                    row.append(round(get_expected_value(state_dist)[0], 2))
                elif isinstance(state_dist, HalfNormal):
                    row.append(state_dist.covs.numpy().item())
                elif isinstance(state_dist, StudentT):
                    row.append(state_dist.dofs.numpy().item())
                    row.append(state_dist.means.numpy().item())
                    row.append(state_dist.covs.numpy().item())
            else:
                for feature_idx, dist in enumerate(state_dist.distributions):
                    feature_name = list(self.features.keys())[feature_idx]
                    try:
                        dist_type = DistributionType(self.features[feature_name])
                    except ValueError:
                        raise ValueError(f"Unsupported distribution type: {self.features[feature_name]}. Supported types are: {list(DistributionType)}")
                    if isinstance(dist, StudentT):
                        row.append(dist.dofs.numpy().item())
                        row.append(dist.means.numpy().item())
                        row.append(dist.covs.numpy().item())
                    elif isinstance(dist, Normal):
                        row.extend([dist.means.numpy().item(), dist.covs.numpy().item()])
                    elif isinstance(dist, Poisson):
                        row.append(dist.lambdas.numpy().item())
                    elif isinstance(dist, Categorical):
                        row.append(dist.probs.numpy()[0].tolist())
                        row.append(round(get_expected_value(dist), 2))
                    elif isinstance(dist, Bernoulli):
                        row.append(dist.probs.numpy().item())
                    elif isinstance(dist, Gamma):
                        row.append(dist.shapes.numpy().item())
                        row.append(dist.rates.numpy().item())
                        row.append(round(get_expected_value(dist)[0], 2))
                    elif isinstance(dist, HalfNormal):
                        row.append(dist.covs.numpy().item())
                    else:
                        raise ValueError(f"Unsupported distribution type: {type(dist)}. Supported types are: {list(DistributionType)}")
            data.append(row)

        # Create the DataFrame with MultiIndex columns
        return pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))

    def get_transition_matrix(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        transition_matrix = self.model.edges.detach().cpu().numpy()
        normalized_transition_matrix = np.exp(transition_matrix) / np.sum(np.exp(transition_matrix), axis=1, keepdims=True)
        return normalized_transition_matrix

    def get_initial_probabilities(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        initial_probabilities = self.model.starts.detach().cpu().numpy()
        normalized_initial_probabilities = normalize_probabilities(initial_probabilities)
        return normalized_initial_probabilities

    def get_end_probabilities(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        end_probabilities = self.model.ends.detach().cpu().numpy()
        normalized_end_probabilities = normalize_probabilities(end_probabilities)
        return normalized_end_probabilities

    def define_hidden_state_labels(self, feature_define_state_labels: Optional[str] = None, labels: Optional[list[str]] = None):
        feature_define_state_labels = feature_define_state_labels or self.config.get("feature_define_state_labels")
        labels = labels or self.config.get("hidden_state_labels")
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if feature_define_state_labels not in self.features.keys():
            raise ValueError(f"Feature '{feature_define_state_labels}' not found in model features.")
        if labels is not None:
            if len(labels) != self.n_states:
                raise ValueError(f"Number of labels ({len(labels)}) must match the number of states ({self.n_states}).")
        else:
            num_sub_categories = int(self.n_states // 3)
            high_labels = ["high_" + str(i+1) for i in range(num_sub_categories)]
            medium_labels = ["medium_" + str(i+1) for i in range(num_sub_categories)]
            remaining_labels = self.n_states - len(high_labels) - len(medium_labels)
            low_labels = ["low_" + str(i+1) for i in range(remaining_labels)]
            labels = low_labels + medium_labels + high_labels
            self.config['hidden_state_labels'] = labels

        feature_define_state_labels_dists = self.feature_dist[feature_define_state_labels]
        expected_values = []
        for state_dist in feature_define_state_labels_dists:
            expected_value = np.mean(get_expected_value(state_dist)) # expected value is a vector
            expected_values.append(expected_value)

        expected_values = np.array(expected_values)
        sorted_indices = np.argsort(expected_values)
        self.hidden_state_labels = {sorted_indices[i].item(): labels[i] for i in range(len(labels))}

    def get_hidden_state_labels(self):
        if self.hidden_state_labels is None:
            raise ValueError("Hidden state labels have not been defined. Please call define_hidden_state_labels() first.")
        return self.hidden_state_labels

    def calculate_aic(self, df) -> float:
        """
        Compute AIC (Akaike Information Criterion) for the model on the given data.
        """
        log_likelihood, _ = self.log_likelihood(df, total=True)
        k = self.get_total_num_params()
        aic = 2 * k - 2 * log_likelihood        
        return aic

    def calculate_bic(self, df) -> float:
        """
        Compute BIC (Bayesian Information Criterion) for the model on the given data.
        BIC = log(N) * k - 2 * log(L)
        where:
        - N is the number of sequences (or observations)
        - k is the number of free parameters in the model
        - L is the likelihood of the model given the data
        """
        log_likelihood, num_sequences = self.log_likelihood(df, total=True)
        k = self.get_total_num_params()
        # BIC = log(N) * k - 2 * log(L)
        bic = np.log(num_sequences) * k - 2 * log_likelihood
        return bic.item()

    def get_total_num_params(self) -> int:
        """
        Estimate the number of free parameters in the HMM:
        - Transition matrix: n_states * (n_states - 1)
        - Initial probabilities: n_states - 1
        - Emission distributions: call model-specific helper (or hard-code if fixed)
        """
        num_transitions = self.n_states * (self.n_states - 1)
        num_initial_probs = self.n_states - 1
        num_emissions = self._num_emission_params()
        return num_transitions + num_initial_probs + num_emissions

    def _num_emission_params(self) -> int:
        """
        Calculate the number of emission distribution parameters for the HMM.
        Assumes all features are univariate and defined in self.features as:
        {feature_name: distribution_type}
        """

        total_params_per_state = 0

        for feature_name, dist_type in self.features.items():
            try:
                dist_type = DistributionType(dist_type)
            except ValueError:
                raise ValueError(f"Unsupported distribution type: {dist_type}. Supported types are: {list(DistributionType)}")
            if dist_type == DistributionType.POISSON:
                total_params_per_state += 1
            elif dist_type in [DistributionType.GAUSSIAN, DistributionType.NORMAL, DistributionType.GAMMA, DistributionType.LOG_NORMAL]:
                total_params_per_state += 2  # mean + variance (or shape + scale for gamma)
            elif dist_type == DistributionType.STUDENT_T:
                total_params_per_state += 2  # mean + variance (dof is treates as hyperparameter - fixed)
            elif dist_type == DistributionType.CATEGORICAL:
                num_categories = self.feature_dist[feature_name][0].probs.shape[1]
                if num_categories is None:
                    raise ValueError(f"Number of categories for '{feature_name}' not found.")
                total_params_per_state += (num_categories - 1)  # last one is implied
            elif dist_type == DistributionType.GAMMA:
                total_params_per_state += 2  # shape + scale
            elif dist_type == DistributionType.HALF_NORMAL:
                total_params_per_state += 1  # scale
            elif dist_type == DistributionType.EXPONENTIAL:
                total_params_per_state += 1  # rate
            elif dist_type == DistributionType.BERNOULLI:
                total_params_per_state += 1  # probability
            else:
                raise ValueError(f"Unsupported distribution type: {dist_type}")

        return self.n_states * total_params_per_state

    def save_model(self, model_path: Optional[str] = None):
        """
        Save the trained model to a file.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call train() first.")

        if model_path is None:
            model_path = self.config.get('save_model_path', './_models')

        model_path = model_path + "/" + self.config.get('model_name', 'hmm')

        os.makedirs(model_path, exist_ok=True)

        model_path = model_path + "/" + self.config.get('model_name', 'hmm') + '.pkl'

        print(f"Saving model to {model_path}")

        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

        model_config_path = model_path.replace('.pkl', '_config.yaml')

        print(f"Saving model config to {model_config_path}")

        with open(model_config_path, 'w') as f:
            yaml.dump(self.config, f)

        return model_path

    def save_model_params(self, model_path: Optional[str] = None):
        """
        Save the model parameters to a txt file.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call train() first.")

        if model_path is None:
            model_path = self.config.get('save_model_path', './_models')

        model_path = model_path + "/" + self.config.get('model_name', 'hmm')

        os.makedirs(model_path, exist_ok=True)

        file_path = model_path + "/" + self.config.get('model_name', 'hmm') + '_params.txt'

        print(f"Saving model parameters to {file_path}")

        with open(file_path, 'w') as f:
            params_str = self.get_model_params_str()
            f.write(params_str)

    def save_model_metrics(self, df, model_path: Optional[str] = None):
        """
        Save the model metrics (AIC, BIC, etc.) to a .txt file.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call train() first.")
        if model_path is None:
            model_path = self.config.get('save_model_path', './_models')

        model_path = model_path + "/" + self.config.get('model_name', 'hmm')
        os.makedirs(model_path, exist_ok=True)

        file_path = model_path + "/" + self.config.get('model_name', 'hmm') + '_metrics.txt'

        aic = self.calculate_aic(df)
        bic = self.calculate_bic(df)
        log_likelihood, num_samples = self.log_likelihood(df, total=True)

        print(f"Saving model metrics to {file_path}")
        with open(file_path, 'w') as f:
            f.write(f"Log Likelihood: {round(log_likelihood, 2)}\n")
            f.write(f"Number of Sequences: {num_samples}\n")
            f.write(f"AIC: {round(aic, 2)}\n")
            f.write(f"BIC: {round(bic, 2)}\n")
            f.write(f"Number of States: {self.n_states}\n")
            f.write(f"Number of Parameters: {self.get_total_num_params()}\n")

def get_model_selection_results_hmm(
    max_num_states, config, X, initialize_from_first_obs_with_gmm, n_repeats = 5, max_attempts_per_fit=5
):
    """
    For each number of hidden states, fit the model n_repeat times, and return mean and std of AIC/BIC.
    """
    all_aic_runs = []
    all_bic_runs = []
    aic_mean_values = []
    aic_std_values = []
    bic_mean_values = []
    bic_std_values = []
    num_params_list = []
    num_states_list = list(range(2, max_num_states + 1))

    print(f"Training HMM models with max number of hidden states: {max_num_states}")
    features = config.get("features", {})
    print("Features: ", features)
    np.random.RandomState(seed=56)

    if X[list(features.keys())].isna().any().any():
        raise ValueError("Input data contains NaN values. Please handle missing values before fitting the model.")
    
    apply_power_transform = config.get("apply_power_transform", False)
    print(f"Apply power transform: {apply_power_transform}")

    for n_states in tqdm(num_states_list):
        aic_runs = []
        bic_runs = []
        params_runs = []

        for run in range(n_repeats):
            print(f"Run {run + 1}/{n_repeats} for n_states={n_states}")
            hmm = RecurrentHealthEventsHMM(
                config=config,
                n_states=n_states
            )
            success = False
            for attempt in range(max_attempts_per_fit):
                try:
                    random_state = np.random.randint(0, 10000)
                    if apply_power_transform:
                        X = hmm.fit_transform_power_variables(X)
                    hmm.train(
                        X,
                        verbose=False, 
                        random_state=random_state, 
                        initialize_from_first_obs_with_gmm=initialize_from_first_obs_with_gmm
                    )
                    success = True
                    break
                except Exception as e:
                    print(f"Run {run+1}, attempt {attempt+1} failed: {e}")
            if not success:
                print(f"Failed to fit model for n_states={n_states}, run={run+1}")
                continue  # Skip this run

            # Get params, aic, bic for this run
            params_runs.append(hmm.get_total_num_params())
            aic_runs.append(hmm.calculate_aic(X))
            bic_runs.append(hmm.calculate_bic(X))

        if len(aic_runs) == 0:
            print(f"No successful fits for n_states={n_states}. Skipping.")
            continue

        num_params_list.append(np.mean(params_runs).item())
        all_aic_runs.append(aic_runs)
        all_bic_runs.append(bic_runs)
        aic_mean_values.append(np.nanmean(aic_runs).item())
        aic_std_values.append(np.nanstd(aic_runs).item())
        bic_mean_values.append(np.nanmean(bic_runs).item())
        bic_std_values.append(np.nanstd(bic_runs).item())

    return {
        "num_states": num_states_list[:len(aic_mean_values)],
        "num_params": num_params_list,
        "aic_mean": aic_mean_values,
        "aic_std": aic_std_values,
        "bic_mean": bic_mean_values,
        "bic_std": bic_std_values,
        "aic_runs": all_aic_runs,
        "bic_runs": all_bic_runs
    }
