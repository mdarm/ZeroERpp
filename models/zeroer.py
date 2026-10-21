"""Model layer — the pure ZeroER EM model (the "network").

This is the model concern of the three-layer ml-project-style split: the
unsupervised-matching model and nothing else — no pyJedAI stage glue, no
blocking, no data loading, no experiment plumbing. It depends only on
numpy/scipy/scikit-learn/pandas, so it could be lifted into a completely
different project unchanged.

The default EM path follows upstream ZeroER's ``model.py``
(https://github.com/chu-data-lab/zeroer). Local extensions support explicit
initial labels, missing attributes, diagnostics and held-out density scoring.

Exports
-------
ZeroerModel
    Two-component Gaussian-mixture EM over a similarity-feature matrix;
    ``ZeroerModel.run_em(...)`` is the fit loop (E-step / M-step until the
    free-energy convergence test trips), with optional transitivity.
get_y_init_given_threshold
    Threshold-on-scaled-sum bootstrap labelling that seeds EM.
ConvergenceMeter
    Free-energy convergence test used inside ``run_em``.
bay_coeff, getScaledSum
    Pure helpers used inside the M-step / initialisation.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import newton
from scipy.special import expit
from scipy.stats import multivariate_normal, norm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from tqdm.autonotebook import tqdm

__all__ = [
    "ZeroerModel",
    "get_y_init_given_threshold",
    "ConvergenceMeter",
    "bay_coeff",
    "getScaledSum",
]

DEL = 1e-300


def getScaledSum(similarity_features):
    feature_sums = np.sum(similarity_features, axis=1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feature_sums.reshape(-1, 1))
    return scaled


def get_y_init_given_threshold(similarity_features_df, threshold=0.8):
    x = similarity_features_df.values
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    scaled_sum = getScaledSum(x_scaled)
    training_labels_ = scaled_sum > threshold
    y_init = [int(val) for val in training_labels_.flatten()]
    return y_init


def _blocks(feature_names):
    """{attribute block: its column indices}, grouped as ``ZeroerModel`` groups them."""
    blocks = defaultdict(list)
    for i_col, name in enumerate(feature_names):
        blocks[name.split("_")[0]].append(i_col)
    return blocks


def initial_labels(X, feature_names, threshold=0.8, statistic="sum", present=None):
    """The bootstrap labels that seed EM. Not upstream code, except ``"sum"``.

    ``"sum"`` is upstream's ``get_y_init_given_threshold``: the scaled sum of all
    columns. It counts columns, so an attribute with more features weighs more,
    and an absent attribute (a zero block) pulls a pair down — on sparse data the
    seed is a split on *presence*. ``"present_mean"`` averages within each block
    and then across the blocks that are present for the pair (``present``:
    {block: boolean array}; None = all present), which removes both effects.
    """
    if statistic == "sum":
        return get_y_init_given_threshold(pd.DataFrame(X), threshold)
    if statistic != "present_mean":
        raise ValueError("init statistic must be 'sum' or 'present_mean'")
    x_scaled = MinMaxScaler().fit_transform(np.asarray(X, dtype=float))
    blocks = _blocks(feature_names)
    means = np.column_stack([x_scaled[:, cols].mean(axis=1) for cols in blocks.values()])
    there = np.ones_like(means, dtype=bool) if present is None else \
        np.column_stack([np.asarray(present[block], dtype=bool) for block in blocks])
    statistic = (means * there).sum(axis=1) / np.maximum(there.sum(axis=1), 1)
    scaled = MinMaxScaler().fit_transform(statistic.reshape(-1, 1))
    return [int(val) for val in (scaled > threshold).flatten()]


def _get_results(true_labels, predicted_labels):
    p = precision_score(true_labels, predicted_labels)
    r = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    return p, r, f1


def bay_coeff(a, b, u):
    return np.exp(-(np.log(a / (b + DEL) + b / (a + DEL) + 2) + u / (a + b + DEL)))


class ConvergenceMeter:
    def __init__(self, num_converged, rate_threshold,
                 diff_fn=lambda a, b: abs(a - b)):
        self._num_converged = num_converged
        self._rate_threshold = rate_threshold
        self._diff_fn = diff_fn
        self._diff_history = list()
        self._last_val = None

    def offer(self, val):
        if self._last_val is not None:
            self._diff_history.append(
                self._diff_fn(val, self._last_val))

        self._last_val = val

    @property
    def is_converged(self):
        if len(self._diff_history) < self._num_converged:
            return False

        return np.mean(
            self._diff_history[-self._num_converged:]) \
            <= self._rate_threshold


class ZeroerModel:
    """
    Pairwise similarity features
            ↓
    Initial rough labels (from a threshold on similarity)
            ↓
    Start match / non-match model
            ↓
    Repeat EM steps:
    
        E-step:
            estimate probability of match
    
        → optional consistency fix:
            ensure matches are logically consistent (transitivity)
    
        M-step:
            update model parameters
            (means, variances, covariances)
    
        → smooth / regularize covariance updates
    
        → check if results have stabilized (convergence test)
    
            ↓
    Final output:
        probability each pair is a match

    """
    class Gaussian:
        def __init__(self, mu, std):
            self.mu = mu
            self.std = (std + DEL)

        def pdf(self, s):
            return norm.pdf(s, loc=self.mu, scale=self.std)

        def logpdf(self, s):
            return norm.logpdf(s, loc=self.mu, scale=self.std)

    def __init__(self, similarity_matrix, feature_names, y, id_df, c_bay,
                 pi_M=None, hard=False, init_threshold=0.8, init_statistic="sum",
                 present=None):
        self.c_bay = c_bay
        self.X = np.array(similarity_matrix)
        if y is None:
            y = initial_labels(self.X, feature_names, init_threshold,
                               init_statistic, present)
        self.y = np.asarray(y)
        if self.y.ndim != 1 or len(self.y) != len(self.X):
            raise ValueError("initial labels must have one value per feature row")
        if not np.isin(self.y, (0, 1)).all():
            raise ValueError("initial labels must be binary (0 or 1)")
        self.y = self.y.astype(int, copy=True)
        if np.any(np.bincount(self.y, minlength=2) < 2):
            raise ValueError("initial labels need at least two rows in each class")
        self.id_tuple_to_index = {}
        if id_df is not None:
            self.ids = id_df.values
            for i in range(self.ids.shape[0]):
                self.id_tuple_to_index[(self.ids[i, 0], self.ids[i, 1])] = i
                self.id_tuple_to_index[(self.ids[i, 1], self.ids[i, 0])] = i

        Mu_all = np.mean(self.X, axis=0)
        self.Cov_all = np.dot(np.transpose(self.X - Mu_all), (self.X - Mu_all)) / self.X.shape[0]
        self.corr = pd.DataFrame(similarity_matrix).corr().values
        self.sigma = np.zeros_like(self.corr)
        for i in range(self.corr.shape[0]):
            self.sigma[i, i] = np.std(self.X[:, i])
        self.P_M = np.zeros(self.X.shape[0])  # M is class 1
        self.Q_avg = 0
        self.feature_names = feature_names

        self.col_index_2_group_name = []
        self.group_name_2_col_indices = defaultdict(list)
        for i_col, name in enumerate(feature_names):
            self.col_index_2_group_name.append(name.split("_")[0])
            self.group_name_2_col_indices[self.col_index_2_group_name[-1]].append(i_col)
        self.group_names = list(set(self.col_index_2_group_name))

        # Missing-aware fit — not upstream, and off (None) unless ``present`` =
        # {block: boolean array} is given. An absent attribute is then left out of
        # the likelihood and of every parameter estimate instead of being read
        # as a block of zero similarities.
        self.observed = None
        if present is not None:
            self.observed = np.zeros(self.X.shape, dtype=bool)
            for g_name, i_cols in self.group_name_2_col_indices.items():
                rows = np.asarray(present[g_name], dtype=bool)
                self.observed[:, i_cols] = rows[:, None]
                self.corr[np.ix_(i_cols, i_cols)] = np.nan_to_num(
                    pd.DataFrame(self.X[rows][:, i_cols]).corr().values)

        if pi_M is None:
            pi_M = float(self.y.mean())

        self._hard = hard
        self._num_rows = self.X.shape[0]
        self._num_cols = self.X.shape[1]
        self._labels = list(sorted(np.unique(self.y)))
        self.y_step = self.y.copy()

        self.pi_M = pi_M
        self.pi_M_l = pi_M
        self.pi_M_r = pi_M
        self.params = []
        self.Mu_M = np.zeros((self._num_cols,))
        self.Mu_U = np.zeros((self._num_cols,))
        self.Cov_M = np.zeros((self._num_cols, self._num_cols))
        self.Cov_U = np.zeros((self._num_cols, self._num_cols))
        for i in range(self._num_cols):
            self.params.append(self.fit_conditional_parameters(i))
            self.Mu_U[i] = self.params[-1][0].mu
            self.Mu_M[i] = self.params[-1][1].mu
            self.Cov_U[i, i] = self.params[-1][0].std ** 2
            self.Cov_M[i, i] = self.params[-1][1].std ** 2
        self.P_M_2_dimen = None
        self.log_P_M_2_dimen = None
        self.log_P_U_2_dimen = None

    def get_class_wise_scores(self, i_cols):
        class_wise_scores = dict()
        for label in self._labels:
            class_wise_scores[label] = \
                self.X[np.where(self.y == label), i_cols]
            if self.observed is not None:
                rows = (np.asarray(self.y) == label) & self.observed[:, i_cols]
                if rows.sum() > 1:
                    class_wise_scores[label] = self.X[rows, i_cols]

        return class_wise_scores

    def fit_conditional_parameters(self, i):
        class_wise_scores = self.get_class_wise_scores(i)

        class_wise_parameters = dict()
        for label in self._labels:
            gmm = GaussianMixture(n_components=1)
            gmm.fit(class_wise_scores[label].reshape(-1, 1))

            class_wise_parameters[label] = \
                self.Gaussian(mu=gmm.means_.flatten()[0],
                              std=np.sqrt(gmm.covariances_.flatten()[0]))

        return class_wise_parameters

    def e_step(self, model_l=None, model_r=None):
        self.model_l = model_l
        self.model_r = model_r

        reg_cov = 1e-8 * np.identity(len(self.X[0]))
        self.Cov_M += reg_cov
        self.Cov_U += reg_cov

        min_eig = np.min(np.real(np.linalg.eigvals(self.Cov_M)))
        if min_eig < 0:
            self.Cov_M -= 10 * min_eig * np.eye(*self.Cov_M.shape)
        min_eig = np.min(np.real(np.linalg.eigvals(self.Cov_U)))
        if min_eig < 0:
            self.Cov_U -= 10 * min_eig * np.eye(*self.Cov_U.shape)
        if self.observed is None:
            log_prods_dup = multivariate_normal.logpdf(self.X, mean=self.Mu_M, cov=self.Cov_M, allow_singular=True)
            log_prods_non_dup = multivariate_normal.logpdf(self.X, mean=self.Mu_U, cov=self.Cov_U, allow_singular=True)
        else:
            log_prods_dup = self._observed_logpdf(self.Mu_M, self.Cov_M)
            log_prods_non_dup = self._observed_logpdf(self.Mu_U, self.Cov_U)

        pi_M = self.pi_M
        pi_U = 1 - pi_M

        prob_non_dup_over_dup = np.exp(np.clip(log_prods_non_dup - log_prods_dup, -500, 500))

        self.Q_M = log_prods_dup
        self.Q_U = log_prods_non_dup

        self.P_M = pi_M / (pi_M + pi_U * prob_non_dup_over_dup)
        self.P_U = 1 - self.P_M
        if self._hard:
            self.P_M = np.round(np.clip(self.P_M, 0., 1.))

    def _observed_logpdf(self, mu, cov):
        """``log N(x; mu, cov)`` over each pair's *present* blocks. Σ is
        block-diagonal, so the density factorises over blocks and marginalising
        an absent one is exact — it drops out of the sum. A pair with no block
        present scores 0 under both classes: its posterior is the prior."""
        total = np.zeros(self._num_rows)
        for i_cols in self.group_name_2_col_indices.values():
            rows = self.observed[:, i_cols[0]]
            if rows.any():
                total[rows] += multivariate_normal.logpdf(
                    self.X[rows][:, i_cols], mean=mu[i_cols],
                    cov=cov[np.ix_(i_cols, i_cols)], allow_singular=True)
        return total

    def free_energy(self):
        return self.P_M * (np.log(self.pi_M + DEL) - np.log(self.P_M + DEL) + self.Q_M) \
            + self.P_U * (np.log(1 - self.pi_M + DEL) - np.log(self.P_U + DEL) + self.Q_U)

    def expected_log_likelihood(self):
        """Equation (4) of the paper, summed over pairs — the statistic its
        stopping rule watches. Not upstream code: upstream stops on
        ``free_energy`` instead (see ``run_em``'s ``tol``)."""
        return np.sum(self.P_M * (np.log(self.pi_M + DEL) + self.Q_M)
                      + self.P_U * (np.log(1 - self.pi_M + DEL) + self.Q_U))

    def score_components(self, X, present=None):
        """Score new rows under the final fitted parameters without changing them.

        Columns must have the fitted feature order. A missing-aware model needs
        a row-aligned presence mask and marginalises absent attribute blocks;
        a zero-fill model uses every block, even if a mask is supplied.
        ``p_match`` is the generative probability, before hard assignment or
        transitivity. It can differ from the last E-step's retained posterior.
        ``n_blocks`` counts the blocks contributing to each density.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._num_cols:
            raise ValueError("scoring matrix must have the fitted feature width")
        if not np.isfinite(X).all():
            raise ValueError("scoring matrix must contain only finite values")
        if self.observed is not None and present is None:
            raise ValueError("missing-aware scoring requires present=")
        masks = {}
        for block in self.group_name_2_col_indices:
            if present is not None:
                if block not in present:
                    raise ValueError("presence mask is missing block {!r}".format(block))
                rows = np.asarray(present[block])
                if rows.shape != (len(X),) or not np.isin(rows, (False, True)).all():
                    raise ValueError("presence mask for {!r} must contain one boolean "
                                     "per scoring row".format(block))
                masks[block] = rows.astype(bool)
            if self.observed is None:
                masks[block] = np.ones(len(X), dtype=bool)
        n_blocks = np.zeros(len(X), dtype=int)
        for rows in masks.values():
            n_blocks += rows

        def logpdf(mu, covariance):
            # Match the E-step's stabilisation, using fresh local arrays.
            covariance = covariance.copy() + 1e-8 * np.eye(self._num_cols)
            min_eig = np.min(np.real(np.linalg.eigvals(covariance)))
            if min_eig < 0:
                covariance -= 10 * min_eig * np.eye(self._num_cols)
            result = np.zeros(len(X))
            if self.observed is None:
                if len(X):
                    result[:] = multivariate_normal.logpdf(
                        X, mean=mu, cov=covariance, allow_singular=True)
            else:
                for block, cols in self.group_name_2_col_indices.items():
                    rows = masks[block]
                    if rows.any():
                        result[rows] += multivariate_normal.logpdf(
                            X[rows][:, cols], mean=mu[cols],
                            cov=covariance[np.ix_(cols, cols)], allow_singular=True)
            return result

        pi = float(self.pi_M)
        if not np.isfinite(pi) or not 0 <= pi <= 1:
            raise ValueError("fitted match prior must lie between 0 and 1")
        log_match = logpdf(self.Mu_M, self.Cov_M)
        log_nonmatch = logpdf(self.Mu_U, self.Cov_U)
        llr = log_match - log_nonmatch
        log_pi = np.log(pi) if pi > 0 else -np.inf
        log_pi_U = np.log1p(-pi) if pi < 1 else -np.inf
        log_density = np.logaddexp(log_pi + log_match, log_pi_U + log_nonmatch)
        p_match = expit(log_pi - log_pi_U + llr)
        blind = n_blocks == 0
        log_density[blind] = 0.
        p_match[blind] = pi
        return dict(log_match=log_match, log_nonmatch=log_nonmatch, llr=llr,
                    p_match=p_match, log_density=log_density,
                    n_blocks=n_blocks, pi=pi)

    def enforce_transitivity(self, P_M, ids, id_tuple_to_index, model_l, model_r,
                             LR_dup_free=False, LR_identical=False):
        model_l_P_M = None
        model_r_P_M = None
        if model_l is not None:
            model_l_P_M = model_l.P_M
            model_r_P_M = model_r.P_M
            id_tuple_to_index_l = model_l.id_tuple_to_index
            id_tuple_to_index_r = model_r.id_tuple_to_index
        P_M = P_M.copy()
        pred_tuples = []

        for i in range(P_M.shape[0]):
            if P_M[i] > 0.5:
                pred_tuples.append((ids[i, 0], ids[i, 1]))
        pred_tuples = sorted(pred_tuples)

        for i in range(len(pred_tuples)):
            for j in range(i + 1, len(pred_tuples)):
                if pred_tuples[j][0] == pred_tuples[i][0]:
                    p1 = P_M[id_tuple_to_index[pred_tuples[i]]]
                    p2 = P_M[id_tuple_to_index[pred_tuples[j]]]
                    p_r = 0
                    id1 = id_tuple_to_index[pred_tuples[i]]
                    id2 = id_tuple_to_index[pred_tuples[j]]
                    if LR_dup_free:
                        p_r = 0
                        idr = -1
                    elif LR_identical:
                        if (pred_tuples[i][1], pred_tuples[j][1]) not in id_tuple_to_index:
                            p_r = 0
                            idr = -1
                        else:
                            p_r = P_M[id_tuple_to_index[(pred_tuples[i][1], pred_tuples[j][1])]]
                            idr = id_tuple_to_index[(pred_tuples[i][1], pred_tuples[j][1])]
                    elif model_r_P_M is not None:
                        if (pred_tuples[i][1], pred_tuples[j][1]) not in id_tuple_to_index_r:
                            p_r = 0
                            idr = -1
                        else:
                            p_r = model_r_P_M[id_tuple_to_index_r[(pred_tuples[i][1], pred_tuples[j][1])]]
                            idr = id_tuple_to_index_r[(pred_tuples[i][1], pred_tuples[j][1])]

                    if p1 * p2 > p_r:
                        delta_ls = [self.delta_L(p_r / p2, id1), self.delta_L(p_r / p1, id2)]
                        if idr != -1:
                            if LR_identical:
                                delta_ls.append(self.delta_L(p1 * p2, idr))
                            else:
                                delta_ls.append(model_r.delta_L(p1 * p2, idr))
                        i_max = np.argmax(delta_ls)
                        if delta_ls[i_max] > -1e100:
                            if i_max == 0:
                                P_M[id1] = p_r / p2
                            elif i_max == 1:
                                P_M[id2] = p_r / p1
                            elif i_max == 2:
                                if LR_identical:
                                    P_M[idr] = p1 * p2
                                else:
                                    model_r_P_M[idr] = p1 * p2
                else:
                    break

        pred_tuples = sorted(pred_tuples, key=lambda x: (x[1], x[0]))
        for i in range(len(pred_tuples)):
            for j in range(i + 1, len(pred_tuples)):
                if pred_tuples[j][1] == pred_tuples[i][1]:
                    p1 = P_M[id_tuple_to_index[pred_tuples[i]]]
                    p2 = P_M[id_tuple_to_index[pred_tuples[j]]]
                    p_l = 0
                    id1 = id_tuple_to_index[pred_tuples[i]]
                    id2 = id_tuple_to_index[pred_tuples[j]]
                    if LR_dup_free:
                        p_l = 0
                        idl = -1
                    elif LR_identical:
                        if (pred_tuples[i][0], pred_tuples[j][0]) not in id_tuple_to_index:
                            p_l = 0
                            idl = -1
                        else:
                            p_l = P_M[id_tuple_to_index[(pred_tuples[i][0], pred_tuples[j][0])]]
                            idl = id_tuple_to_index[(pred_tuples[i][0], pred_tuples[j][0])]
                    elif model_l_P_M is not None:
                        if (pred_tuples[i][0], pred_tuples[j][0]) not in id_tuple_to_index_l:
                            p_l = 0
                            idl = -1
                        else:
                            p_l = model_l_P_M[id_tuple_to_index_l[(pred_tuples[i][0], pred_tuples[j][0])]]
                            idl = id_tuple_to_index_l[(pred_tuples[i][0], pred_tuples[j][0])]
                    if p1 * p2 > p_l:
                        delta_ls = [self.delta_L(p_l / p2, id1), self.delta_L(p_l / p1, id2)]
                        if idl != -1:
                            if LR_identical:
                                delta_ls.append(self.delta_L(p1 * p2, idl))
                            else:
                                delta_ls.append(model_l.delta_L(p1 * p2, idl))
                        i_max = np.argmax(delta_ls)
                        if delta_ls[i_max] > -1e100:
                            if i_max == 0:
                                P_M[id1] = p_l / p2
                            elif i_max == 1:
                                P_M[id2] = p_l / p1
                            elif i_max == 2:
                                if LR_identical:
                                    P_M[idl] = p1 * p2
                                else:
                                    model_l_P_M[idl] = p1 * p2
                else:
                    break
        if model_r_P_M is not None:
            model_l.P_M = model_l_P_M
            model_r.P_M = model_r_P_M
        return P_M

    def m_step(self):
        N = self._num_rows

        X = self.X
        P_M = self.P_M
        P_U = 1. - P_M

        if self._hard:
            P_M = P_M.astype(int)
            P_U = P_U.astype(int)

        N_M = np.sum(P_M, axis=0)
        N_U = N - N_M

        self.pi_M = N_M / N

        P_M = P_M.reshape(N, 1)
        P_U = P_U.reshape(N, 1)

        if self.observed is not None:
            Cov_M, std_M = self._observed_moments(P_M, "Mu_M", self.Cov_M)
            Cov_U, std_U = self._observed_moments(P_U, "Mu_U", self.Cov_U)
            smooth_factor = abs((self.Mu_M - self.Mu_U)) ** 2
        else:
            self.Mu_M = np.sum(P_M * X, axis=0) / (N_M + DEL)
            self.Mu_U = np.sum(P_U * X, axis=0) / (N_U + DEL)

            smooth_factor = abs((self.Mu_M - self.Mu_U)) ** 2

            std_M = (np.sqrt(np.sum(
                P_M * ((X - np.tile(self.Mu_M, (N, 1))) ** 2), axis=0) / (N_M + DEL))) + 1e-100
            std_U = (np.sqrt(np.sum(
                P_U * ((X - np.tile(self.Mu_U, (N, 1))) ** 2), axis=0) / (N_U + DEL))) + 1e-100

            Cov_M = np.dot(np.transpose(self.X - self.Mu_M), P_M * (self.X - self.Mu_M)) / (N_M + DEL)
            Cov_U = np.dot(np.transpose(self.X - self.Mu_U), P_U * (self.X - self.Mu_U)) / (N_U + DEL)

        a = np.diag(Cov_M)
        b = np.diag(Cov_U)
        u = (self.Mu_M - self.Mu_U) ** 2
        c = 0.15

        c_bay = self.c_bay
        bay_ori = bay_coeff(a, b, u)
        target_bay = bay_ori + c_bay
        target_bay[target_bay >= 1] = bay_ori[target_bay >= 1] / 2 + 0.5

        def bay_coeff_equ(x):
            return bay_coeff(a + x, b + x, u) - target_bay

        x0 = c * smooth_factor
        x1 = np.zeros_like(x0)
        kappas = newton(bay_coeff_equ, x0=x0, x1=x1, maxiter=5, tol=1)
        kappas[kappas < 0] = 0
        kappas[kappas > 1] = 1
        kappas = np.nan_to_num(kappas, posinf=0, neginf=0)
        self.kappas = kappas  # kept for diagnostics; upstream discards it
        self.bay_ori = bay_ori
        self.Cov_M = np.zeros_like(Cov_M)
        self.Cov_U = np.zeros_like(Cov_U)

        for g_name in self.group_names:
            i_cols = self.group_name_2_col_indices[g_name]

            for col_1 in i_cols:
                for col_2 in i_cols:
                    if col_2 == col_1:
                        self.Cov_M[col_1, col_2] = Cov_M[col_1, col_2] + kappas[col_1]
                        self.Cov_U[col_1, col_2] = Cov_U[col_1, col_2] + kappas[col_1]
                    else:
                        self.Cov_M[col_1, col_2] = self.corr[col_1, col_2] * std_M[col_1] * std_M[col_2]
                        self.Cov_U[col_1, col_2] = self.corr[col_1, col_2] * std_U[col_1] * std_U[col_2]

    def _observed_moments(self, P, mu_name, previous_cov):
        """One class's means and variances with every column weighted by its own
        observed rows (the missing-aware M-step); returns (diagonal covariance,
        std). A column with no observed mass in the class keeps what it had."""
        W = P * self.observed
        mass = W.sum(axis=0)
        seen = mass > 1e-8
        mu = np.where(seen, (W * self.X).sum(axis=0) / (mass + DEL), getattr(self, mu_name))
        var = np.where(seen, (W * (self.X - mu) ** 2).sum(axis=0) / (mass + DEL),
                       np.diag(previous_cov))
        setattr(self, mu_name, mu)
        return np.diag(var), np.sqrt(var) + 1e-100

    def L(self, q, i):
        return q * (np.log(self.pi_M + DEL) + self.Q_M[i] - np.log(q + DEL)) \
            + (1 - q) * (np.log(1 - self.pi_M + DEL) + self.Q_U[i] - np.log(1 - q + DEL))

    def delta_L(self, q, i):
        delta = self.L(q, i) - self.L(self.P_M[i], i)
        if delta > 0.00001:
            return -1e200
        return delta

    @classmethod
    def run_em(cls, similarity_matrixs, feature_names, y_inits, id_dfs,
               LR_dup_free, LR_identical, run_trans,
               c_bay=0.015,
               y_true=None,
               pi_M=None,
               hard=False,
               max_iter=40,
               init_threshold=0.8,
               tol=None,
               init_statistic="sum",
               present=None,
               trace=False):
        """``init_threshold`` and ``tol`` are the two knobs upstream hard-wires.
        Their defaults are upstream's behaviour: initial labels from the scaled
        feature sum > 0.8, and a stop when the norm of the change in the per-pair
        ``free_energy`` vector averages <= 0.01 over ten iterations. A ``tol``
        switches to the paper's rule (section 3.3): stop when
        ``expected_log_likelihood`` — Equation (4), read after the E-step, where
        posteriors and parameters agree — moves by less than ``tol`` between two
        consecutive iterations. The returned model carries ``n_iter`` and
        ``converged`` (False = stopped by ``max_iter``).

        ``init_statistic`` (see ``initial_labels``) and ``present`` (the
        missing-aware fit, see ``__init__``) apply to the cross-table model only.
        ``trace`` keeps, per iteration, the posteriors and the regulariser's state
        in ``model.history`` — for diagnosis; it changes nothing."""
        sims, sims_l, sims_r = similarity_matrixs
        y_init, y_init_l, y_init_r = y_inits
        id_df, id_df_l, id_df_r = id_dfs
        model = cls(sims, feature_names, y_init, id_df, pi_M=pi_M, hard=hard, c_bay=c_bay,
                    init_threshold=init_threshold, init_statistic=init_statistic,
                    present=present)
        model.history = []
        if run_trans and LR_dup_free is False and LR_identical is False:
            model_l = cls(sims_l, feature_names, y_init_l, id_df_l, c_bay=c_bay,
                          init_threshold=init_threshold)
            model_r = cls(sims_r, feature_names, y_init_r, id_df_r, c_bay=c_bay,
                          init_threshold=init_threshold)

        if tol is None:
            convergence = ConvergenceMeter(10, 0.01, diff_fn=lambda a, b: np.linalg.norm(a - b))
        else:
            convergence = ConvergenceMeter(1, tol)
        model.converged = False

        with tqdm(range(max_iter)) as pbar:
            for i in pbar:
                model.n_iter = i + 1
                model.e_step()
                if run_trans:
                    if LR_dup_free is False and LR_identical is False:
                        model_r.e_step()
                        model_l.e_step()
                    for i in range(4):
                        if LR_dup_free is False and LR_identical is False:
                            model_l.P_M = model_l.enforce_transitivity(
                                model_l.P_M, model_l.ids, model_l.id_tuple_to_index, model_l, model_l)
                            model_r.P_M = model_r.enforce_transitivity(
                                model_r.P_M, model_r.ids, model_r.id_tuple_to_index, model_r, model_r)
                            model.P_M = model.enforce_transitivity(
                                model.P_M, model.ids, model.id_tuple_to_index, model_l, model_r)
                        else:
                            model.P_M = model.enforce_transitivity(
                                model.P_M, model.ids, model.id_tuple_to_index, None, None,
                                LR_dup_free, LR_identical)
                if tol is not None:
                    expected_ll = model.expected_log_likelihood()
                model.m_step()
                if run_trans and LR_dup_free is False and LR_identical is False:
                    model_r.m_step()
                    model_l.m_step()

                if trace:
                    model.history.append(dict(P_M=model.P_M.copy(), pi_M=model.pi_M,
                                              kappas=model.kappas.copy(),
                                              bay_ori=model.bay_ori.copy()))
                convergence.offer(model.free_energy() if tol is None else expected_ll)
                if convergence.is_converged:
                    model.converged = True
                    break
                if y_true is not None:
                    y_pred = np.round(np.clip(model.P_M + DEL, 0., 1.)).astype(int) \
                        if not hard else model.P_M.astype(int)
                    p, r, f1 = _get_results(y_true, y_pred)
                    result_str = (
                        "norm: {:0.2f}, "
                        "F1: {:0.2f}, "
                        "Precision: {:0.2f}, "
                        "Recall: {:0.2f}".format(
                            np.linalg.norm(model.P_M),
                            f1, p, r))
                    pbar.set_description_str(result_str)

        return model, model.P_M
