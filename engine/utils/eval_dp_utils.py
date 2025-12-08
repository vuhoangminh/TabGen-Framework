import time
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import entropy, multivariate_normal, wasserstein_distance

# Custom packages/modules
from engine.datasets import get_dataset

try:
    from anonymeter.evaluators import (
        SinglingOutEvaluator,
        LinkabilityEvaluator,
        InferenceEvaluator,
    )
except:
    pass

from sdmetrics.single_table import CategoricalZeroCAP, CategoricalGeneralizedCAP


# ===========================================================================
# ML methods
# ---------------------------------------------------------------------------
# Privacy Metrics - perhaps
# Check -- Privacy Loss Budget (ε): Measures the allowable information leakage from the dataset.
# Check -- Expected Loss: Evaluates the potential loss associated with predictions made using synthetic data compared to real data.
# Check -- Global Sensitivity Analysis (Lipschitz continuity): Quantifies how sensitive the output (synthetic data) is to changes in the real data by using the Lipschitz property in a DP framework. Lower sensitivity means the synthetic data is sufficiently different from the real data.
# Check -- Differential Privacy Guarantee (Bounded Sensitivity): By bounding the sensitivity of queries, this method ensures that changes in any single record in the real dataset have a limited impact on the output of the synthetic data, quantifying privacy preservation.

# Privacy Metrics - must
# Try this https://nbviewer.org/github/vanderschaarlab/synthcity/blob/main/tutorials/tutorial5_differential_privacy.ipynb https://github.com/vanderschaarlab/synthcity/tree/main


def compute_categoricalcap(df_real, df_fake, key_fields, sensitive_fields):
    """
    CategoricalCAP The CategoricalCAP measures the risk of disclosing sensitive information through an inference attack. We assume that some values in the real data are public knowledge. An attacker is combining this with synthetic data to make guesses about other real values that are sensitive. This metric describes how difficult it is for an attacker to correctly guess the sensitive information using an algorithm called Correct Attribution Probability (CAP)

    Source:
    - modified from
    from https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/categoricalcap
    """

    try:
        score_zero = CategoricalZeroCAP.compute(
            real_data=df_real,
            synthetic_data=df_fake,
            key_fields=key_fields,
            sensitive_fields=sensitive_fields,
        )

        score_general = CategoricalGeneralizedCAP.compute(
            real_data=df_real,
            synthetic_data=df_fake,
            key_fields=key_fields,
            sensitive_fields=sensitive_fields,
        )

        return {
            "dp_categorical_zero_cap": score_zero,  # higher is better
            "dp_categorical_generalized_cap": score_general,  # higher is better
        }

    except:
        return {
            "dp_categorical_zero_cap": np.nan,
            "dp_categorical_generalized_cap": np.nan,
        }


def compute_dcr_nndr(real, fake, data_percent=15):
    """
    Returns privacy metrics

    Inputs:
    1) real_path -> path to real data
    2) fake_path -> path to corresponding synthetic data
    3) data_percent -> percentage of data to be sampled from real and synthetic datasets for computing privacy metrics

    Outputs:
    1) List containing the 5th percentile distance to closest record (DCR) between real and synthetic as well as within real and synthetic datasets
    along with 5th percentile of nearest neighbour distance ratio (NNDR) between real and synthetic as well as within real and synthetic datasets

    Source:
    - modified from
    https://github.com/Team-TUD/CTAB-GAN/blob/main/model/eval/evaluation.py
    """
    # Sampling smaller sets of real and synthetic data to reduce the time complexity of the evaluation
    real_sampled = real.sample(
        n=int(len(real) * (0.01 * data_percent)),
        # random_state=42,
    ).to_numpy()
    fake_sampled = fake.sample(
        n=int(len(fake) * (0.01 * data_percent)),
        # random_state=42,
    ).to_numpy()

    # Scaling real and synthetic data samples
    scalerR = StandardScaler()
    scalerR.fit(real_sampled)
    scalerF = StandardScaler()
    scalerF.fit(fake_sampled)
    df_real_scaled = scalerR.transform(real_sampled)
    df_fake_scaled = scalerF.transform(fake_sampled)

    # Computing pair-wise distances between real and synthetic
    dist_rf = metrics.pairwise_distances(
        df_real_scaled, Y=df_fake_scaled, metric="minkowski", n_jobs=-1
    )
    # # Computing pair-wise distances within real
    # dist_rr = metrics.pairwise_distances(
    #     df_real_scaled, Y=None, metric="minkowski", n_jobs=-1
    # )
    # # Computing pair-wise distances within synthetic
    # dist_ff = metrics.pairwise_distances(
    #     df_fake_scaled, Y=None, metric="minkowski", n_jobs=-1
    # )

    # # Removes distances of data points to themselves to avoid 0s within real and synthetic
    # rd_dist_rr = dist_rr[~np.eye(dist_rr.shape[0], dtype=bool)].reshape(
    #     dist_rr.shape[0], -1
    # )
    # rd_dist_ff = dist_ff[~np.eye(dist_ff.shape[0], dtype=bool)].reshape(
    #     dist_ff.shape[0], -1
    # )

    # Computing first and second smallest nearest neighbour distances between real and synthetic
    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [
        dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))
    ]
    # # Computing first and second smallest nearest neighbour distances within real
    # smallest_two_indexes_rr = [
    #     rd_dist_rr[i].argsort()[:2] for i in range(len(rd_dist_rr))
    # ]
    # smallest_two_rr = [
    #     rd_dist_rr[i][smallest_two_indexes_rr[i]] for i in range(len(rd_dist_rr))
    # ]
    # # Computing first and second smallest nearest neighbour distances within synthetic
    # smallest_two_indexes_ff = [
    #     rd_dist_ff[i].argsort()[:2] for i in range(len(rd_dist_ff))
    # ]
    # smallest_two_ff = [
    #     rd_dist_ff[i][smallest_two_indexes_ff[i]] for i in range(len(rd_dist_ff))
    # ]

    # Computing 5th percentiles for DCR and NNDR between and within real and synthetic datasets
    min_dist_rf = np.array([i[0] for i in smallest_two_rf])
    fifth_perc_rf = np.percentile(min_dist_rf, 5)
    # min_dist_rr = np.array([i[0] for i in smallest_two_rr])
    # fifth_perc_rr = np.percentile(min_dist_rr, 5)
    # min_dist_ff = np.array([i[0] for i in smallest_two_ff])
    # fifth_perc_ff = np.percentile(min_dist_ff, 5)
    nn_ratio_rf = np.array([i[0] / i[1] for i in smallest_two_rf])
    nn_fifth_perc_rf = np.percentile(nn_ratio_rf, 5)
    # nn_ratio_rr = np.array([i[0] / i[1] for i in smallest_two_rr])
    # nn_fifth_perc_rr = np.percentile(nn_ratio_rr, 5)
    # nn_ratio_ff = np.array([i[0] / i[1] for i in smallest_two_ff])
    # nn_fifth_perc_ff = np.percentile(nn_ratio_ff, 5)

    return np.array(
        [
            fifth_perc_rf,
            # fifth_perc_rr,
            # fifth_perc_ff,
            nn_fifth_perc_rf,
            # nn_fifth_perc_rr,
            # nn_fifth_perc_ff,
        ]
    )


def bootstrap_compute_dcr_nndr(
    real, fake, data_percent=15, num_bootstrap=100, num_workers=-1
):
    """
    Perform bootstrapping on the privacy_metrics function using parallelization with joblib.

    Parameters:
    - real: pandas DataFrame containing real data.
    - fake: pandas DataFrame containing synthetic data.
    - data_percent: percentage of data to sample (default is 15).
    - num_bootstrap: number of bootstrap rounds (default is 1000).
    - num_workers: number of parallel workers to use (default is -1, which uses all available cores).

    Returns:
    - results: numpy array of shape (num_bootstrap, 6) containing the bootstrapped results.
    """

    # Function to compute a single bootstrap iteration
    def bootstrap_iteration(_):
        return compute_dcr_nndr(real, fake, data_percent)

    try:
        # Parallelize the bootstrap iterations
        results = Parallel(n_jobs=num_workers)(
            delayed(bootstrap_iteration)(i) for i in range(num_bootstrap)
        )

        # Convert results to numpy array
        results = np.array(results)

        # Compute mean and standard deviation across all bootstrap rounds
        mean_results = np.mean(results, axis=0)

        return {
            "dp_dcr_rf": mean_results[0],  # higher is better
            "dp_nndr_rf": mean_results[1],  # higher is better
        }

    except:
        return {
            "dp_dcr_rf": np.nan,
            "dp_nndr_rf": np.nan,
        }


def compute_single_out_risk(
    df_real,
    df_fake,
    df_hold,
    n_attacks=500,
    list_n_cols=[1, 2, 4, 8],
):
    """
    Parameters
    ----------
    n_attacks : int
        Total number of attacks performed.
    n_cols : int, default is 3
        Number of columns that the attacker uses to create the singling
        out queries. -> input case by case
    n_success : int
        Number of successful attacks.
    n_baseline : int
        Number of successful attacks for the
        baseline (i.e. random-guessing) attacker.
    n_control : int, default is None
        Number of successful attacks against the
        control dataset. If this parameter is not None
        the privacy risk will be measured relative to
        the attacker success on the control set.
    confidence_level : float, default is 0.95
        Desired confidence level for the confidence
        intervals on the risk.

    Source:
    - modified from
    https://github.com/statice/anonymeter/tree/main/src/anonymeter/evaluators
    """

    risks = {}

    for n_cols in list_n_cols:
        if n_cols == 1:
            mode = "univariate"
            max_attempts = 10_000_000
        else:
            mode = "multivariate"
            max_attempts = 10_000

        print(f">> processing single out {n_cols}")
        evaluator = SinglingOutEvaluator(
            ori=df_real,
            syn=df_fake,
            control=df_hold,
            n_attacks=n_attacks,
            n_cols=n_cols,
            max_attempts=max_attempts,
        )

        try:
            evaluator.evaluate(mode=mode)
            risk = evaluator.risk()
            print(risk)
            res = evaluator.results()
            print("Successs rate of main attack:", res.attack_rate)
            print("Successs rate of baseline attack:", res.baseline_rate)
            print("Successs rate of control attack:", res.control_rate)
            risks.update(
                {
                    f"dp_single_out_risk_{n_cols}": risk.value,  # smaller is better
                    f"dp_single_out_train_vs_control_{n_cols}": res.attack_rate.value
                    - res.control_rate.value,  # smaller is better
                    f"dp_single_out_train_vs_naive_{n_cols}": res.attack_rate.value
                    - res.baseline_rate.value,  # smaller is better
                }
            )

        except:
            risks.update(
                {
                    f"dp_single_out_risk_{n_cols}": np.nan,
                    f"dp_single_out_train_vs_control_{n_cols}": np.nan,
                    f"dp_single_out_train_vs_naive_{n_cols}": np.nan,
                }
            )

    return risks


# check how to use
def compute_linkability_risk(
    df_real,
    df_fake,
    df_hold,
    aux_cols,
    n_attacks=5_000,
    list_n_neighbors=[4, 5, 6, 7, 8, 9, 10],
):
    """
    ori : pd.DataFrame
        Dataframe containing original data.
    syn : pd.DataFrame
        Dataframe containing synthetic data. It has to have
        the same columns as df_ori.
    aux_cols : tuple of two lists of strings or tuple of int, optional
        Features of the records that are given to the attacker as auxiliary
        information.
    n_attacks : int, default is 500.
        Number of records to attack. If None each record in the original
        dataset will be attacked.
    n_neighbors : int, default is 1
        The number of closest neighbors to include in the analysis. The
        default of 1 means that the linkability attack is considered
        successful only if the two original record split have the same
        synthetic record as closest neighbor.

    Source:
    - modified from
    https://github.com/statice/anonymeter/tree/main/src/anonymeter/evaluators
    """
    risks = {}

    try:
        evaluator = LinkabilityEvaluator(
            ori=df_real,
            syn=df_fake,
            control=df_hold,
            n_attacks=n_attacks,
            aux_cols=aux_cols,
            n_neighbors=10,
        )

        evaluator.evaluate(n_jobs=-2)
        for n_neighbors in list_n_neighbors:
            risks[f"dp_linkability_{n_neighbors}"] = evaluator.risk(
                n_neighbors=n_neighbors
            ).value  # smaller is better
    except:
        for n_neighbors in list_n_neighbors:
            risks[f"dp_linkability_{n_neighbors}"] = np.nan

    return risks


def compute_inference_risk(
    df_real,
    df_fake,
    df_hold,
    key_fields,
    sensitive_fields,
    n_attacks=5_000,
):
    """
    Source:
    - modified from
    https://github.com/statice/anonymeter/tree/main/src/anonymeter/evaluators
    """
    risks = {}

    for secret in sensitive_fields:
        aux_cols = key_fields
        try:
            evaluator = InferenceEvaluator(
                ori=df_real,
                syn=df_fake,
                control=df_hold,
                aux_cols=aux_cols,
                secret=secret,
                n_attacks=n_attacks,
            )
            evaluator.evaluate(n_jobs=-2)

            risks[f"dp_inference_{secret}"] = (
                evaluator.risk().value
            )  # smaller is better
        except:
            risks[f"dp_inference_{secret}"] = np.nan

    return risks


def get_features(X: pd.DataFrame, sensitive_features: List[str] = []) -> List[str]:
    """Return the non-sensitive features from dataset X."""
    features = list(X.columns)
    for col in sensitive_features:
        if col in features:
            features.remove(col)
    return features


def compute_k_anonymization(
    df_real: pd.DataFrame, df_synthetic: pd.DataFrame, key_features: List[str]
) -> Tuple[int, int]:
    """
    Compute k-Anonymity for real and synthetic datasets.

    Parameters:
    - df_real: DataFrame containing the real dataset
    - df_synthetic: DataFrame containing the synthetic dataset
    - sensitive_features: List of sensitive features to exclude from k-anonymity evaluation

    Returns:
    - k_real: k-anon value for real data
    - k_synthetic: k-anon value for synthetic data

    Source:
    - modified from
    https://github.com/vanderschaarlab/synthcity/blob/6043e1b4de2836eac7a378a0c464260b421c5064/src/synthcity/metrics/eval_privacy.py
    """

    def evaluate_data(X: pd.DataFrame) -> int:
        features = get_features(X, key_features)
        values = [999]  # Initialize with a high value

        # Test various cluster sizes
        for n_clusters in [2, 5, 10, 15]:
            if len(X) / n_clusters < 10:
                continue  # Skip if too few samples per cluster
            cluster = KMeans(
                n_clusters=n_clusters, init="k-means++", random_state=0
            ).fit(X[features])
            counts: dict = Counter(cluster.labels_)
            values.append(np.min(list(counts.values())))

        return int(np.min(values))

    try:
        # Calculate k-anonymity for both datasets
        k_real = evaluate_data(df_real)
        k_synthetic = evaluate_data(df_synthetic)

        return {
            "dp_k_anonymization_synthetic": k_synthetic,  # higher is better
            "dp_k_anonymization_safe": (
                1 if k_synthetic >= k_real else 0
            ),  # higher is better
        }
    except:
        return {
            "dp_k_anonymization_synthetic": np.nan,
            "dp_k_anonymization_safe": np.nan,
        }


def compute_l_diversity_distinct(
    df_real: pd.DataFrame,
    df_fake: pd.DataFrame,
    key_features: list,
    n_clusters_list: list = [2, 5, 10, 15],
) -> Dict[str, float]:
    """
    Source:
    - modified from
    https://github.com/vanderschaarlab/synthcity/blob/6043e1b4de2836eac7a378a0c464260b421c5064/src/synthcity/metrics/eval_privacy.py
    """

    def evaluate_data(df: pd.DataFrame) -> int:
        features = get_features(df, key_features)
        values = [999]
        for n_clusters in n_clusters_list:
            if len(df) / n_clusters < 10:
                continue
            model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0).fit(
                df[features]
            )
            clusters = model.predict(df[features])
            clusters_df = pd.Series(clusters, index=df.index)

            for cluster in range(n_clusters):
                partition = df[clusters_df == cluster]
                uniq_values = partition[key_features].drop_duplicates()
                values.append(len(uniq_values))

        return int(np.min(values))

    try:
        real_distinct = evaluate_data(df_real)
        fake_distinct = evaluate_data(df_fake)

        # Return the l-Diversity results
        return {
            "dp_l_diversity_synthetic": fake_distinct,  # higher is better
            "dp_l_diversity_safe": (
                1 if fake_distinct >= real_distinct else 0
            ),  # higher is better
        }
    except:
        return {
            "dp_l_diversity_synthetic": np.nan,
            "dp_l_diversity_safe": np.nan,
        }


def compute_k_map(
    df_real: pd.DataFrame,
    df_fake: pd.DataFrame,
    key_features: list,
    n_clusters_list: list = [2, 4, 8, 16],
) -> Dict[str, int]:
    """
    Source:
    - modified from
    https://github.com/vanderschaarlab/synthcity/blob/6043e1b4de2836eac7a378a0c464260b421c5064/src/synthcity/metrics/eval_privacy.py
    """

    # Ensure the sensitive features exist in the DataFrames
    try:
        features = get_features(df_real, key_features)
        values = []
        for n_clusters in n_clusters_list:
            if len(df_real) / n_clusters < 10:
                continue

            # Fit KMeans to the real data
            model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0).fit(
                df_real[features]
            )
            clusters = model.predict(df_fake[features])

            # Count occurrences of each cluster in the synthetic data
            counts = Counter(clusters)

            # Find the minimum count of the clusters corresponding to the real data
            values.append(np.min(list(counts.values())))

        # Handle cases where no clusters are valid
        if len(values) == 0:
            return 0

        return {
            "dp_k_map": int(np.min(values)),  # higher is better
        }

    except:
        return {
            "dp_k_map": np.nan,
        }


def compute_delta_presence(
    df_real: pd.DataFrame,
    df_fake: pd.DataFrame,
    sensitive_features: List[str],
    n_clusters_list: list = [2, 4, 8, 16],
) -> Dict[str, float]:
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.DeltaPresence
        :parts: 1

    Returns the maximum re-identification probability on the real dataset from the synthetic dataset.
    For each dataset partition, we report the maximum ratio of unique sensitive information between the real dataset and in the synthetic dataset.

    Source:
    - modified from
    https://github.com/vanderschaarlab/synthcity/blob/6043e1b4de2836eac7a378a0c464260b421c5064/src/synthcity/metrics/eval_privacy.py
    """

    try:
        features = get_features(df_real, sensitive_features)
        values = []
        for n_clusters in n_clusters_list:
            if len(df_real) / n_clusters < 10:
                continue

            # Cluster the real dataset
            model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0).fit(
                df_real[features]
            )
            clusters_real = model.labels_  # Get labels for the real dataset

            # Predict clusters for the synthetic dataset
            clusters_fake = model.predict(df_fake[features])

            # Count occurrences in each cluster
            synth_counts = Counter(clusters_fake)
            gt_counts = Counter(clusters_real)

            # Calculate the maximum ratio of counts
            for key in gt_counts:
                if key not in synth_counts:
                    continue
                gt_cnt = gt_counts[key]
                synth_cnt = synth_counts[key]

                delta = gt_cnt / (synth_cnt + 1e-8)  # Avoid division by zero
                values.append(delta)

        return {
            "dp_delta_presence": (
                float(np.max(values)) if values else np.nan
            ),  # smaller is better
        }
    except:
        return {
            "dp_delta_presence": np.nan,
        }


def compute_re_identification(
    df_real: pd.DataFrame, df_fake: pd.DataFrame
) -> Dict[str, float]:
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.IdentifiabilityScore
        :parts: 1

    Returns the re-identification score on the real dataset from the synthetic dataset.

    We estimate the risk of re-identifying any real data point using synthetic data.
    Intuitively, if the synthetic data are very close to the real data, the re-identification risk would be high.
    The precise formulation of the re-identification score is given in the reference below.

    Reference: Jinsung Yoon, Lydia N. Drumright, Mihaela van der Schaar,
    "Anonymization through Data Synthesis using Generative Adversarial Networks (ADS-GAN):
    A harmonizing advancement for AI in medicine,"
    IEEE Journal of Biomedical and Health Informatics (JBHI), 2019.
    Paper link: https://ieeexplore.ieee.org/document/9034117

    Source:
    - modified from
    https://github.com/vanderschaarlab/synthcity/blob/6043e1b4de2836eac7a378a0c464260b421c5064/src/synthcity/metrics/eval_privacy.py
    """

    def compute_scores(
        X_gt: np.ndarray, X_syn: np.ndarray, emb: str = ""
    ) -> Dict[str, float]:
        """Calculate the re-identification score."""

        def compute_entropy(labels: np.ndarray) -> float:
            """Compute the entropy of the given labels."""
            value, counts = np.unique(np.round(labels), return_counts=True)
            return entropy(counts)

        # Reshape if necessary
        X_gt_ = X_gt.reshape(len(X_gt), -1)
        X_syn_ = X_syn.reshape(len(X_syn), -1)

        # Entropy computation
        no, x_dim = X_gt_.shape
        W = np.zeros(x_dim)

        for i in range(x_dim):
            W[i] = compute_entropy(X_gt_[:, i])

        # Normalization
        eps = 1e-16
        for i in range(x_dim):
            X_gt_[:, i] /= W[i] + eps
            X_syn_[:, i] /= W[i] + eps

        # Nearest neighbors computation
        nbrs = NearestNeighbors(n_neighbors=2).fit(X_gt_)
        distance, _ = nbrs.kneighbors(X_gt_)

        nbrs_hat = NearestNeighbors(n_neighbors=1).fit(X_syn_)
        distance_hat, _ = nbrs_hat.kneighbors(X_gt_)

        # Identifiability value computation
        R_Diff = distance_hat[:, 0] - distance[:, 1]
        identifiability_value = np.sum(R_Diff < 0) / float(no)

        return identifiability_value

    # Convert DataFrames to numpy arrays
    X_gt = df_real.to_numpy()
    X_syn = df_fake.to_numpy()

    try:
        # Evaluate scores without the one-class model
        result = compute_scores(X_gt, X_syn)

        return {
            "dp_re_identification_score": result,  # smaller is better
        }
    except:
        return {
            "dp_re_identification_score": np.nan,
        }


def compute_domiasmia(
    df_test,  # Evaluation dataset
    df_fake,  # Synthetic dataset
    df_real,  # Dataset used to create the mem_set
    synth_val_set,  # Dataset for density evaluation
    mode,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate various Membership Inference Attacks.

    Args:
        X_gt: The evaluation dataset used to derive training and test datasets.
        synth_set: The synthetic dataset.
        X_train: The dataset used to create the mem_set.
        synth_val_set: The dataset for calculating the density of the synthetic data.
        reference_size: The size of the reference dataset.
        device: The device to be used (CPU or CUDA).

    Returns:
        A dictionary with accuracy and AUCROC scores for the attack.

    Source:
    - modified from
    https://github.com/vanderschaarlab/synthcity/blob/6043e1b4de2836eac7a378a0c464260b421c5064/src/synthcity/metrics/eval_privacy.py
    """

    class normal_func_feat:
        def __init__(
            self,
            X: np.ndarray,
            continuous: list,
        ) -> None:
            if np.any(np.array(continuous) > 1) or len(continuous) != X.shape[1]:
                raise ValueError("Continous variable needs to be boolean")
            self.feat = np.array(continuous).astype(bool)

            if np.sum(self.feat) == 0:
                raise ValueError("there needs to be at least one continuous feature")

            self.var = np.std(X[:, self.feat], axis=0) ** 2
            self.mean = np.mean(X[:, self.feat], axis=0)

        def pdf(self, Z: np.ndarray) -> np.ndarray:
            return multivariate_normal.pdf(
                Z[:, self.feat], self.mean, np.diag(self.var)
            )

    def compute_metrics_baseline(
        y_scores: np.ndarray,
        y_true: np.ndarray,
        sample_weight=None,
    ) -> Tuple[float, float]:
        y_pred = y_scores > np.median(y_scores)
        y_true = np.nan_to_num(y_true)
        y_pred = np.nan_to_num(y_pred)
        y_scores = np.nan_to_num(y_scores)
        acc = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        auc = roc_auc_score(y_true, y_scores, sample_weight=sample_weight)
        return acc, auc

    def evaluate_p_R_prior(
        synth_set, synth_val_set, reference_set, X_test, norm
    ) -> Tuple[np.ndarray, np.ndarray]:
        density_gen = stats.gaussian_kde(synth_set.values.transpose(1, 0))
        p_G_evaluated = density_gen(X_test.transpose(1, 0))
        p_R_evaluated = norm.pdf(X_test)
        return p_G_evaluated, p_R_evaluated

    def evaluate_p_R_kde(
        synth_set, synth_val_set, reference_set, X_test, norm
    ) -> Tuple[np.ndarray, np.ndarray]:

        density_gen = stats.gaussian_kde(synth_set.values.transpose(1, 0))
        density_data = stats.gaussian_kde(reference_set.transpose(1, 0))
        p_G_evaluated = density_gen(X_test.transpose(1, 0))
        p_R_evaluated = density_data(X_test.transpose(1, 0))
        return p_G_evaluated, p_R_evaluated

    try:
        reference_size = int(len(df_test.index) * 0.7)
        df_real = df_real.to_numpy()
        df_test = df_test.to_numpy()

        # Prepare member and non-member sets
        mem_set = df_real  # Assuming X_train is already a numpy array
        non_mem_set, reference_set = (
            df_test[:reference_size],
            df_test[-reference_size:],
        )

        # Combine real datasets
        all_real_data = np.concatenate((df_real, df_test), axis=0)

        # Determine if features are continuous
        continuous = []
        for i in range(all_real_data.shape[1]):
            continuous.append(0 if len(np.unique(all_real_data[:, i])) < 10 else 1)

        # Normalization function
        norm = normal_func_feat(all_real_data, continuous)  # Implement this function

        # Prepare test sets for members and non-members
        X_test = np.concatenate([mem_set, non_mem_set])
        Y_test = np.concatenate(
            [np.ones(mem_set.shape[0]), np.zeros(non_mem_set.shape[0])]
        ).astype(bool)

        # Density estimation
        if mode == "prior":
            evaluate_p_R = evaluate_p_R_prior
        elif mode == "kde":
            evaluate_p_R = evaluate_p_R_kde
        else:
            raise ValueError

        p_G_evaluated, p_R_evaluated = evaluate_p_R(
            df_fake, synth_val_set, reference_set, X_test, norm
        )

        # Relative probability calculation
        p_rel = p_G_evaluated / (p_R_evaluated + 1e-10)

        # Compute metrics
        acc, auc = compute_metrics_baseline(p_rel, Y_test)  # Implement this function

        return {
            "dp_domias_mia_accuracy": acc,  # smaller is better
            "dp_domias_mia_auc": auc,  # smaller is better
        }

    except:
        return {
            "dp_domias_mia_accuracy": np.nan,
            "dp_domias_mia_auc": np.nan,
        }


# ---------------------------------------------------------------------------
# ML methods - end
# ===========================================================================


# ===========================================================================
# main functions - start
# ---------------------------------------------------------------------------
def main():
    D = get_dataset("adult")
    df_hold = D.data_test
    df_real = D.data_train

    folder, filename_fake = (
        "adult-tvae-lv_2-bs_27000-epochs_1400-ed_32-cd_256-dd_256-l2_7.68e-06-moment_1-losscorcorr_4.54e+02-lossdis_6.76e-08-condvec_1",
        "fake_01400",
    )

    df_fake = pd.read_csv(
        f"database/gan_optimize/{folder}/{filename_fake}.csv",
        sep="\t",
        header=0,
        index_col=0,
    )

    # Iterate through the columns of df1 and set the dtypes in df2
    for column in df_fake.columns:
        if column in df_real.columns:  # Check if the column exists in df2
            df_real[column] = df_real[column].astype(df_fake[column].dtype)
            df_hold[column] = df_hold[column].astype(df_fake[column].dtype)

    df_test = df_hold.copy()

    key_fields = D.key_fields
    sensitive_fields = D.sensitive_fields
    # sensitive_fields = []

    scores = {}

    # df_real = df_real.sample(frac=0.2, random_state=42)
    # df_fake = df_fake.sample(frac=0.2, random_state=42)
    # df_hold = df_hold.sample(frac=0.2, random_state=42)

    # Step 1: k-anonymization
    start = time.time()
    x = compute_k_anonymization(df_real, df_fake, key_fields)
    print(f"k-anonymization: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 2: l-Diversity
    start = time.time()
    x = compute_l_diversity_distinct(df_real, df_fake, key_fields)
    print(f"l-Diversity: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 3: k-map
    start = time.time()
    x = compute_k_map(df_real, df_fake, key_fields)
    print(f"k-map: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 4: Delta presence
    start = time.time()
    x = compute_delta_presence(df_real, df_fake, key_fields)
    print(f"delta_presence: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 5: Evaluate re-identification
    start = time.time()
    x = compute_re_identification(df_real, df_fake)
    print(f"re_identification: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 6: Domiasmia
    start = time.time()
    x = compute_domiasmia(df_test, df_fake, df_real, df_fake.sample(frac=0.2), "prior")
    print(f"domiasmia prior: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 7: Categorical CAP
    start = time.time()
    x = compute_categoricalcap(df_real, df_fake, key_fields, sensitive_fields)
    print(f"categoricalcap: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 8: DCR Nndr
    start = time.time()
    x = bootstrap_compute_dcr_nndr(df_real, df_fake, data_percent=15, num_bootstrap=100)
    print(f"dcr nndr: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 9: Single-out risk (univariate)
    start = time.time()
    x = compute_single_out_risk(
        df_real,
        df_fake,
        df_hold,
    )
    print(f"single_out_risk: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 11: Linkability risk
    start = time.time()
    x = compute_linkability_risk(
        df_real,
        df_fake,
        df_hold,
        aux_cols=(key_fields, sensitive_fields),
        n_attacks=len(df_hold.index),
    )
    print(f"linkability_risk: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 12: Inference risk
    start = time.time()
    x = compute_inference_risk(
        df_real,
        df_fake,
        df_hold,
        key_fields,
        sensitive_fields,
        n_attacks=len(df_hold.index),
    )
    print(f"inference_risk: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    import pprint

    print()
    print()
    print()
    print()
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(scores)


# ---------------------------------------------------------------------------
# main functions - end
# ===========================================================================
if __name__ == "__main__":
    main()
