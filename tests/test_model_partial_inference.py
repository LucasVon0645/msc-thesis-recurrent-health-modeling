import torch
import numpy as np
import pytest
import pandas as pd
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal, IndependentComponents

from recurrent_health_events_prediction.model.utils import predict_partial_obs_given_history_proba, predict_single_obs_proba, predict_single_partial_obs_proba

@pytest.fixture
def simple_hmm():
    dists = [
        IndependentComponents([
            Normal([0.0], [[1.0]]),  # State 0: mean 0
            Normal([1.0], [[1.0]])   # State 0: mean 1
        ]),
        IndependentComponents([
            Normal([5.0], [[1.0]]),  # State 1: mean 5
            Normal([6.0], [[1.0]])   # State 1: mean 6
        ])
    ]
    model = DenseHMM()
    model.add_distributions(dists)
    model.starts = torch.tensor([0.6, 0.4])
    model.edges = torch.log(torch.tensor([
        [0.9, 0.1],
        [0.1, 0.9]
    ]))
    return model

def test_partial_obs_functions_vs_forward_pomegranate(simple_hmm):
    model = simple_hmm
    # Step 1: Easy sequence (length 2), both features observed
    past_sequence = [
        [0.1, 1.0],   # should be likely under state 0
        [5.2, 7.2]    # should be likely under state 1
    ]
    # "Partial obs": we make two emissions known (no partial observation actuualy), the results should be the same as if we had
    # used the forward algorithm of pomegranate with the full sequence.
    current_obs = [0.2, 1.2]

    # 1. Test predict_single_partial_obs_proba (no history)
    single_probs = predict_single_partial_obs_proba(model, current_obs)
    assert np.isclose(sum(single_probs), 1.0), "Probs should sum to 1"

    # 2. Test predict_partial_obs_given_history_proba (uses history)
    history_probs = predict_partial_obs_given_history_proba(model, past_sequence, current_obs)
    assert np.isclose(sum(history_probs), 1.0), "Probs should sum to 1"

    # 3. Reference: Forward algorithm for [past_sequence + [current_obs]]
    full_seq = past_sequence + [current_obs]
    with torch.no_grad():
        log_alpha = model.forward(torch.tensor([full_seq], dtype=torch.float32)).squeeze()
        last_log_alpha = log_alpha[-1]
        history_probs_ref = torch.exp(last_log_alpha - last_log_alpha.logsumexp(dim=0)).numpy()

    # 4.
    single_probs_ref = predict_single_obs_proba(model, current_obs)

    # Check if predict_partial_obs_given_history_proba outputs the same as forward of pomegranate
    np.testing.assert_allclose(history_probs, history_probs_ref, rtol=1e-5, atol=1e-7)

    # Check if predict_single_partial_obs_proba outputs the same as predict_single_obs_proba
    np.testing.assert_allclose(single_probs, single_probs_ref, rtol=1e-5, atol=1e-7)

def normal_logpdf(x, mu, sigma2):
    """Return log probability of x under Normal(mu, sigma2)."""
    return -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * ((x - mu) ** 2) / sigma2

def manual_likelihood_logspace(model, obs):
    # Use log space throughout!
    priors = model.starts.tolist()  # Already log space!
    n_states = model.n_distributions
    n_features = len(model.distributions[0].distributions)

    emission_logs = []
    for state in range(n_states):
        state_dist = model.distributions[state]
        log_prob = 0.0
        for feat in range(n_features):
            val = obs[feat]
            if not pd.isna(val):
                feature_dist = state_dist.distributions[feat]
                mu = float(getattr(feature_dist, 'mean', feature_dist.means))
                sigma2 = float(getattr(feature_dist, 'covariance', feature_dist.covs)[0][0])
                log_prob += normal_logpdf(val, mu, sigma2)
        emission_logs.append(log_prob)
    joint_log_probs = [prior + emission for prior, emission in zip(priors, emission_logs)]
    # Now normalize via log-sum-exp
    max_log = max(joint_log_probs)
    shifted = [jl - max_log for jl in joint_log_probs]
    exp_shifted = [np.exp(s) for s in shifted]
    sum_exp = sum(exp_shifted)
    manual_probs = [e / sum_exp for e in exp_shifted]
    return manual_probs

def test_predict_single_obs_proba_vs_manual(simple_hmm):
    """
   Test the predict_single_obs_proba function and predict_single_partial_obs_proba
   against a manual implementation.

    Args:
        simple_hmm (DenseHMM): A simple HMM model fixture for testing.
    """
    model = simple_hmm

    # Simulate a partial observation: only feature 0 is observed, feature 1 is missing
    current_obs_partial = [2.0, np.nan]

    current_obs_full = [2.0, 3.0]  # Full observation for reference

    # Call your function
    partial_probs = predict_single_partial_obs_proba(model, current_obs_partial)
    assert np.isclose(sum(partial_probs), 1.0), "Should sum to 1"

    full_probs = predict_single_obs_proba(model, current_obs_full)
    assert np.isclose(sum(full_probs), 1.0), "Should sum to 1"

    partial_probs_ref = manual_likelihood_logspace(model, current_obs_partial)
    print("Function partial probs:", partial_probs)
    print("")
    full_probs_ref = manual_likelihood_logspace(model, current_obs_full)
    print("Function full probs:", full_probs)

    np.testing.assert_allclose(partial_probs, partial_probs_ref,rtol=1e-4, atol=1e-6)

    np.testing.assert_allclose(full_probs, full_probs_ref,rtol=1e-4, atol=1e-6)

def emission_logprob(model, state, obs):
    """Compute the total emission log-prob for state and observation (with NaNs handled)."""
    state_dist = model.distributions[state]
    n_features = len(state_dist.distributions)
    log_prob = 0.0
    for feat in range(n_features):
        val = obs[feat]
        if not pd.isna(val):
            feature_dist = state_dist.distributions[feat]
            mu = float(getattr(feature_dist, 'mean', feature_dist.means))
            sigma2 = float(getattr(feature_dist, 'covariance', feature_dist.covs)[0][0])
            log_prob += normal_logpdf(val, mu, sigma2)
    return log_prob

def manual_history_proba(model, past_obs, current_obs_partial):
    n_states = model.n_distributions
    # 1. Compute log alpha for past observation (obs0)
    log_alpha_0 = []
    for s0 in range(n_states):
        log_prior = model.starts[s0].item()
        log_emission = emission_logprob(model, s0, past_obs)
        log_alpha_0.append(log_prior + log_emission)

    # 2. For each next state, compute logsumexp over previous states plus transition
    log_trans = model.edges.numpy()  # Already in log space, shape [n_states, n_states]
    log_alpha_1 = []
    for s1 in range(n_states):
        terms = []
        for s0 in range(n_states):
            terms.append(log_alpha_0[s0] + log_trans[s0, s1])
        # logsumexp for state s1
        max_term = max(terms)
        logsumexp = max_term + np.log(sum(np.exp(t - max_term) for t in terms))
        # Add emission logprob for current partial obs
        log_emission = emission_logprob(model, s1, current_obs_partial)
        log_alpha_1.append(logsumexp + log_emission)

    # 3. Normalize
    max_log = max(log_alpha_1)
    exp_shifted = [np.exp(v - max_log) for v in log_alpha_1]
    norm = sum(exp_shifted)
    probs = [v / norm for v in exp_shifted]
    return probs

def test_predict_partial_obs_given_history_proba_vs_manual(simple_hmm):
    model = simple_hmm
    # History of one fully observed sample
    past_sequence = [[0.5, 1.1]]  # Try values near state 0's means
    current_obs_partial = [5.1, np.nan]  # Only feature 0 observed, close to state 1

    # Your function
    func_probs = predict_partial_obs_given_history_proba(model, past_sequence, current_obs_partial)
    assert np.isclose(sum(func_probs), 1.0)

    # Manual reference
    manual_probs = manual_history_proba(model, past_sequence[0], current_obs_partial)
    print("Function probs:", func_probs)
    print("Manual probs:", manual_probs)
    np.testing.assert_allclose(func_probs, manual_probs, rtol=1e-4, atol=1e-6)
