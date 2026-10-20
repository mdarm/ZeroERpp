"""String-similarity features rebuilt from the downloaded benchmark tables.

This port implements the feature selection, type inference, tokenizers and
similarity functions used by chu-data-lab/zeroer and py_entitymatching 0.4.0.
The tokenizer/similarity primitives come from py_stringmatching; the runtime
needs neither py_entitymatching nor an upstream checkout or supplied features.

The preparation path is:

    data.reference.load_reference_tables
    -> concatenate both tables as object-typed frames
    -> build_zeroer_features(..., drop_zero_variance=False)
    -> remove known leakage blocks and candidate-cell constant columns at fit

Each table must first be cast with ``astype(object)`` before concatenation.
This preserves integer scalars on one side and floating-point scalars on the
other, such as DBLP-Scholar's years. Their string forms remain ``'1995'`` and
``'1995.0'`` when the similarity function converts them; ordinary numeric
concatenation would coerce both sides to floating point and change the result.
``_as_read_csv`` also accepts stringified numeric inputs, while leaving these
preserved numeric scalars intact.

A missing value on either side produces zero for the pair's attribute features.
Other retained conventions include padded bag q-grams, delimiter tokenization,
raw-value exact matches, the original absolute-normalized-difference rule, and
Magellan's feature names. Names also determine the model's covariance blocks.
Dataset-specific field preparation belongs to ``data.reference``: for example,
Buy's manufacturer is appended to its description before feature extraction.

The replication workflow regenerates all feature intermediates locally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import py_stringmatching as sm

# ---------------------------------------------------------------------------
# Magellan machinery constants
# ---------------------------------------------------------------------------

# extract_features' rank for the cross-table type-upgrade rule.
_TYPE_RANK = {'boolean': 1, 'numeric': 2, 'str_eq_1w': 3, 'str_bt_1w_5w': 4,
              'str_bt_5w_10w': 5, 'str_gt_10w': 6, 'un_determined': 7}

# Magellan's full per-type feature lookup (_get_feat_lkp_tbl), in order.
# (sim_function, tokenizer); tokenizer None = whole-string function.
_STR_LONG = [('cosine', 'dlm_dc0'), ('cosine', 'qgm_3'),
             ('dice', 'dlm_dc0'), ('dice', 'qgm_3'),
             ('monge_elkan', 'dlm_dc0'), ('monge_elkan', 'qgm_3'),
             ('overlap_coeff', 'dlm_dc0'), ('overlap_coeff', 'qgm_3'),
             ('jaccard', 'dlm_dc0'), ('jaccard', 'qgm_3')]
_FEAT_LKP = {
    'str_eq_1w': [('affine', None),
                  ('cosine', 'qgm_2'), ('cosine', 'qgm_3'),
                  ('dice', 'qgm_2'), ('dice', 'qgm_3'),
                  ('lev_dist', None), ('lev_sim', None), ('jaro', None),
                  ('jaro_winkler', None), ('exact_match', None),
                  ('smith_waterman', None), ('needleman_wunsch', None),
                  ('monge_elkan', 'qgm_2'), ('monge_elkan', 'qgm_3'),
                  ('overlap_coeff', 'qgm_2'), ('overlap_coeff', 'qgm_3'),
                  ('jaccard', 'qgm_2'), ('jaccard', 'qgm_3')],
    'str_bt_1w_5w': [('affine', None),
                     ('cosine', 'dlm_dc0'), ('cosine', 'qgm_3'),
                     ('dice', 'dlm_dc0'), ('dice', 'qgm_3'),
                     ('lev_dist', None), ('lev_sim', None), ('jaro', None),
                     ('jaro_winkler', None), ('exact_match', None),
                     ('smith_waterman', None), ('needleman_wunsch', None),
                     ('monge_elkan', 'dlm_dc0'), ('monge_elkan', 'qgm_3'),
                     ('overlap_coeff', 'dlm_dc0'), ('overlap_coeff', 'qgm_3'),
                     ('jaccard', 'dlm_dc0'), ('jaccard', 'qgm_3')],
    'str_bt_5w_10w': _STR_LONG,
    'str_gt_10w': _STR_LONG,
    'numeric': [('exact_match', None), ('abs_norm', None), ('lev_dist', None),
                ('lev_sim', None), ('rel_diff', None)],
    'boolean': [('exact_match', None)],
    'un_determined': [],
}

# get_fn_name's short names; anything absent passes through unchanged
# (dice, overlap_coeff, affine, lev_dist, lev_sim).
_SHORT = {'jaccard': 'jac', 'cosine': 'cos', 'monge_elkan': 'mel',
          'needleman_wunsch': 'nmw', 'smith_waterman': 'sw', 'jaro': 'jar',
          'jaro_winkler': 'jwn', 'exact_match': 'exm', 'abs_norm': 'anm',
          'rel_diff': 'rdf'}

# extract_features + gather_similarity_features drop every feature whose
# *name* contains one of these (distance + non-normalised functions).
_DROP_SUBSTRINGS = ('lev_dist', 'rdf', 'aff', 'sw', 'swn', 'nmw')

# Magellan tokenizer defaults: padded q-grams, bag (not set) semantics.
_TOKENIZERS = {
    'qgm_2': sm.QgramTokenizer(qval=2),
    'qgm_3': sm.QgramTokenizer(qval=3),
    'dlm_dc0': sm.DelimiterTokenizer(delim_set=[' ']),
}
_TOKEN_MEASURES = {'cosine': sm.Cosine(), 'dice': sm.Dice(),
                   'jaccard': sm.Jaccard(),
                   'overlap_coeff': sm.OverlapCoefficient(),
                   'monge_elkan': sm.MongeElkan()}
_LEV = sm.Levenshtein()
_JARO = sm.Jaro()
_JW = sm.JaroWinkler()


# ---------------------------------------------------------------------------
# read_csv dtype reconstruction + Magellan type inference
# ---------------------------------------------------------------------------

def _as_read_csv(column: pd.Series) -> pd.Series:
    """Reproduce pandas.read_csv dtype inference on a possibly astype(str)-
    stringified column. No-op for columns already carrying read_csv dtypes."""
    if (pd.api.types.is_numeric_dtype(column)
            or pd.api.types.is_bool_dtype(column)):
        return column
    col = column.mask(column.isin(['nan', '']))  # astype(str) artifacts -> NaN
    non_null = col.dropna()
    if non_null.empty or not non_null.map(lambda v: isinstance(v, str)).all():
        return col
    numeric = pd.to_numeric(col, errors='coerce')
    if numeric.notna().equals(col.notna()):
        return numeric  # int64 when complete + integral, float64 otherwise
    if non_null.isin(['True', 'False']).all():
        return col.map({'True': True, 'False': False})
    return col


def infer_attr_type(column: pd.Series) -> str:
    """Exact port of em.get_attr_types._get_type (run _as_read_csv first)."""
    column = column.dropna()
    type_list = list(set(column.map(type).tolist()))
    if len(type_list) == 0:
        return 'un_determined'
    if len(type_list) > 1:
        raise AssertionError(
            'Column %s qualifies to be more than one type.' % column.name)
    returned_type = type_list[0]
    if returned_type == bool or returned_type == np.bool_:
        return 'boolean'
    elif returned_type == str:
        average_token_len = pd.Series.mean(
            column.str.split().apply(
                lambda x: len(x) if isinstance(x, list) else np.nan))
        if average_token_len == 1:
            return 'str_eq_1w'
        elif average_token_len <= 5:
            return 'str_bt_1w_5w'
        elif average_token_len <= 10:
            return 'str_bt_5w_10w'
        else:
            return 'str_gt_10w'
    else:
        return 'numeric'


# ---------------------------------------------------------------------------
# Magellan sim-function ports (NaN in either value -> NaN, fillna(0) later)
# ---------------------------------------------------------------------------

def _abs_norm(d1, d2):
    try:
        d1, d2 = float(d1), float(d2)
    except ValueError:
        return np.nan
    if d1 == 0.0 and d2 == 0.0:
        return 0
    x = abs(d1 - d2) / max(abs(d1), abs(d2))
    if x <= 10e-5:  # Magellan's snap-to-equal threshold
        x = 0
    return 1.0 - x


def _to_str(v):
    return v if isinstance(v, str) else str(v)  # gh.convert_to_str_unicode


def _compute(sim, tok_key, raw_l, raw_r, null, strs_l, strs_r, toks):
    n = len(raw_l)
    out = np.full(n, np.nan)
    if tok_key is not None:
        measure = _TOKEN_MEASURES[sim]
        toks_l, toks_r = toks[tok_key]
        for i in range(n):
            if not null[i]:
                out[i] = measure.get_raw_score(toks_l[i], toks_r[i])
        return out
    for i in range(n):
        if null[i]:
            continue
        if sim == 'exact_match':
            out[i] = 1 if raw_l[i] == raw_r[i] else 0
        elif sim == 'abs_norm':
            out[i] = _abs_norm(raw_l[i], raw_r[i])
        elif sim == 'lev_sim':
            out[i] = _LEV.get_sim_score(strs_l[i], strs_r[i])
        elif sim == 'jaro':
            out[i] = _JARO.get_raw_score(strs_l[i], strs_r[i])
        elif sim == 'jaro_winkler':
            out[i] = _JW.get_raw_score(strs_l[i], strs_r[i])
        else:  # affine/sw/nmw/lev_dist/rel_diff never survive the name filter
            raise ValueError(sim)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_zeroer_features(pairs, entities: pd.DataFrame, attributes,
                          dataset_limit: int | None = None,
                          drop_zero_variance: bool = True,
                          n_jobs: int = 1) -> pd.DataFrame:
    """
    pairs: (left_row, right_row) positional indices into ``entities``;
        right-table rows are offset by ``dataset_limit``.
    entities: both tables concatenated after each was cast to object dtype,
        preserving the per-side scalar types; for deduplication, one table.
    attributes: attribute names to featurize, in left-table column order
        (upstream featurizes every shared attribute; 'id' is skipped).
    dataset_limit: first row of the right table in ``entities``;
        None = single-table dedup. Needed because upstream types each table
        separately, then upgrades mismatched types to the higher rank.
    n_jobs: joblib workers for the pair loop (-1 = all cores). Per-pair values
        are independent and type inference/constant-drop are computed globally,
        so the parallel result is identical to the serial one.
    """
    if len(pairs) == 0:
        return pd.DataFrame()
    if n_jobs != 1 and len(pairs) >= 2000:
        from joblib import Parallel, delayed, effective_n_jobs
        chunks = np.array_split(np.arange(len(pairs)), effective_n_jobs(n_jobs))
        parts = Parallel(n_jobs=n_jobs)(
            delayed(build_zeroer_features)(
                [pairs[i] for i in chunk], entities, attributes,
                dataset_limit, drop_zero_variance=False)
            for chunk in chunks if len(chunk))
        fm = pd.concat(parts, ignore_index=True)
        if drop_zero_variance:
            fm = fm.loc[:, fm.nunique() > 1]
        return fm
    li, ri = map(np.asarray, zip(*pairs))
    blocks = []
    for attr in attributes:
        if attr == 'id':  # upstream drops id-based features
            continue
        col = _as_read_csv(entities[attr])
        if dataset_limit is None:
            attr_type = infer_attr_type(col)
        else:
            type_l = infer_attr_type(col.iloc[:dataset_limit])
            type_r = infer_attr_type(col.iloc[dataset_limit:])
            attr_type = max(type_l, type_r, key=_TYPE_RANK.get)

        attr_tag = '_'.join(attr.split())
        feats = []  # (column_name, sim, tok_key) surviving the name filter
        for sim, tok_key in _FEAT_LKP[attr_type]:
            name = '_'.join([attr_tag, attr_tag, _SHORT.get(sim, sim)]
                            + ([tok_key, tok_key] if tok_key else []))
            if not any(f in name for f in _DROP_SUBSTRINGS):
                feats.append((name, sim, tok_key))
        if not feats:
            continue

        raw = col.to_numpy()
        raw_l, raw_r = raw[li], raw[ri]
        null = pd.isnull(raw_l) | pd.isnull(raw_r)
        strs = np.array([_to_str(v) for v in raw], dtype=object)
        strs_l, strs_r = strs[li], strs[ri]

        # Tokenize once per unique string value, per needed tokenizer.
        needed = {t for _, _, t in feats if t is not None}
        uniq, inv = np.unique(strs, return_inverse=True)
        toks = {}
        for tok_key in needed:
            tokenize = _TOKENIZERS[tok_key].tokenize
            tok_u = [tokenize(u) for u in uniq]
            toks[tok_key] = ([tok_u[i] for i in inv[li]],
                             [tok_u[i] for i in inv[ri]])

        data = {name: _compute(sim, tok_key, raw_l, raw_r, null,
                               strs_l, strs_r, toks)
                for name, sim, tok_key in feats}
        blocks.append(pd.DataFrame(data))

    if not blocks:
        return pd.DataFrame()
    fm = pd.concat(blocks, axis=1)
    fm.fillna(value=0, inplace=True)
    if drop_zero_variance:
        fm = fm.loc[:, fm.nunique() > 1]
    return fm


if __name__ == '__main__':
    entities = pd.DataFrame({
        'id':    [0, 1, 2, 3, 4, 5],
        'name':  ['Sony Vaio Laptop', 'Sony Viao Laptop', 'Dell XPS 13',
                  'Apple MacBook Pro 16 inch retina display model',
                  'apple macbook pro 16 inch retina display model',
                  'Dell XPS 13'],
        'brand': ['Sony', 'Sony', 'Dell', 'Apple', 'Apple', 'Dell'],
        'price': ['999', '999', '1200', '1999', 'nan', '1200'],
    })
    pairs = [(0, 1), (0, 2), (2, 5), (3, 4), (1, 1)]
    attrs = ['name', 'brand', 'price']

    for a in attrs:
        print(f'{a:6s} -> inferred type: '
              f'{infer_attr_type(_as_read_csv(entities[a]))}')
    print()

    fm = build_zeroer_features(pairs, entities, attrs)
    pd.set_option('display.width', 240, 'display.max_columns', 60)
    print('shape:', fm.shape)
    print(fm.round(3).T)
