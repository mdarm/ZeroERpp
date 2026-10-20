"""Frozen blocking and October 20 matching experiment design.

Configuration selection minimizes record-disjoint validation NLL within a
panel. Gold-based model/rule comparisons and exact threshold optima are report
diagnostics, not part of the label-free configuration-selection procedure.
"""
from copy import deepcopy

from exp.config import LADDER

MODELS = [
    "minilm6", "minilm12", "mpnet", "distilroberta", "st5_xl", "gtr_t5_large",
    "e5_large_v2", "e5_instruct", "bge_large", "gte_large",
    "Qwen_Qwen3-Embedding-4B", "Qwen_Qwen3-Embedding-8B",
    "codefuse-ai_F2LLM-v2-4B", "codefuse-ai_F2LLM-v2-8B", "codefuse-ai_F2LLM-v2-14B",
    "Octen_Octen-Embedding-8B", "nvidia_llama-embed-nemotron-8b",
    "microsoft_harrier-oss-v1-0.6b", "microsoft_harrier-oss-v1-27b",
    "tencent_KaLM-Embedding-Gemma3-12B-2511",
]
DATASETS = list(LADDER)
PRIMARY_CELLS = [
    ("fodors_zagats", "right", 1), ("dblp_acm", "right", 1),
    ("abt_buy", "left", 2), ("amazon_googleproducts", "left", 4),
    ("dblp_scholar", "left", 6),
]
SENSITIVITY_CELLS = [(dataset, side, k) for dataset, cells in LADDER.items()
                     for side, k in cells]
CELLS = list(dict.fromkeys(PRIMARY_CELLS + SENSITIVITY_CELLS))
SPLIT = dict(n_splits=3, seed=0)
THRESHOLDS = dict(llr=[-5., -2., -1., 0., 1., 2., 5.],
                  posterior=[.1, .25, .5, .75, .9], cosine=[.3, .5, .7, .8, .9])
EXPERIMENT = "unsupervised-v1"
FINAL_EXPERIMENT = "unsupervised-refined-v1"
BASELINE_EXPERIMENT = "unsupervised-baselines-v2"


def _configs(bays):
    return {
        "bay={}:init={}".format(bay, init): dict(
            c_bay=bay, init_statistic=init, init_threshold=.5, missing="ignore",
            hard=False, transitivity=False, pi_M=None, tol=1e-5, max_iter=200)
        for bay in bays for init in ("sum", "present_mean")
    }


CONFIGS = _configs((.001, .01, .1))
EXTRA_CONFIGS = _configs((0., .0001))
PANELS = [dict(name="single:" + model, models=[model], fusion="separate", metrics=["cos"])
          for model in MODELS]
PANELS += [dict(name="all:" + fusion, models=list(MODELS), fusion=fusion, metrics=["cos"])
           for fusion in ("separate", "mean", "zmean", "family")]
PANELS += [
    dict(name="classic:pair", models=["minilm6", "mpnet"], fusion="separate", metrics=["cos"]),
    dict(name="retriever:pair", models=["e5_large_v2", "gte_large"], fusion="separate", metrics=["cos"]),
    dict(name="classic:cos+linf", models=["minilm6", "mpnet"], fusion="separate", metrics=["cos", "linf"]),
]
ABLATIONS = {"zero_fill": dict(missing="zero"), "hard_em": dict(hard=True)}


def select_cells(scope="all", datasets=None):
    pools = {"primary": PRIMARY_CELLS, "sensitivity": SENSITIVITY_CELLS, "all": CELLS}
    if scope not in pools:
        raise ValueError("scope must be primary, sensitivity or all")
    selected = set(DATASETS if datasets is None else datasets)
    unknown = selected - set(DATASETS)
    if unknown or not selected:
        raise ValueError("select known, nonempty datasets; unknown: " + str(sorted(unknown)))
    return [cell for cell in pools[scope] if cell[0] in selected]


def select_panels(names=None):
    if names is None:
        return deepcopy(PANELS)
    names = set(names)
    unknown = names - {panel["name"] for panel in PANELS}
    if unknown or not names:
        raise ValueError("select known, nonempty panels; unknown: " + str(sorted(unknown)))
    return deepcopy([panel for panel in PANELS if panel["name"] in names])


def panel_models(names=None):
    selected = {model for panel in select_panels(names) for model in panel["models"]}
    return [model for model in MODELS if model in selected]
