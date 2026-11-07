"""Time-based validation and tuning for GBMBlend.

The hidden test is homes sold Jun-Sep 2026, valued by a model trained on sales up to
May 2026: 1 to 4 months ahead. These folds copy that exactly:

    fold   train on sales before   predict sales in
    1      2025-06-01              Jun-Sep 2025      } tuning folds: compare settings here
    2      2025-10-01              Oct 2025-Jan 2026 }
    3      2026-02-01              Feb-May 2026        final fold: run ONCE, at the end

Tune only on folds 1-2. Fold 3 then gives an honest estimate of the hidden-test error,
because no setting was chosen by looking at it. (run.py validate retrains every month,
so it only ever predicts 1 month ahead and will look a little better than the real test.)

Usage:
    python tune.py --track off_market --learners lgbm --trials 20
    python tune.py --track off_market --learners lgbm,catboost --trials 10
    python tune.py --track off_market --learners lgbm,catboost --final best_off_market.json
    python tune.py --track off_market --learners lgbm --ablate-text   # does the remarks help?
"""
import argparse
import json
import random
import time
from functools import partial

import numpy as np
import pandas as pd

import data
import features
import models

TUNE_CUTOFFS = ["2025-06-01", "2025-10-01"]
FINAL_CUTOFFS = ["2026-02-01"]
HORIZON_MONTHS = 4

# Settings to search. Keys with a dot go to one learner's parameters.
SEARCH_SPACE = {
    "half_life": [6, 9, 12, 18, 24, None],
    "objective": ["l2", "l1", "huber"],
    "clean": [True, False],
    "text_alpha": [1.0, 3.0, 10.0],
    "lgbm_params.num_leaves": [31, 63, 127, 255],
    "lgbm_params.min_child_samples": [10, 30, 60, 120],
    "lgbm_params.learning_rate": [0.02, 0.03, 0.05],
    "lgbm_params.n_estimators": [800, 1500, 2500],
    "lgbm_params.colsample_bytree": [0.5, 0.7, 0.9],
    "catboost_params.depth": [6, 8],
    "catboost_params.learning_rate": [0.04, 0.06, 0.1],
    "catboost_params.l2_leaf_reg": [1, 3, 10],
    "hgb_params.max_leaf_nodes": [31, 63, 127],
    "hgb_params.min_samples_leaf": [20, 50, 100],
}


# --------------------------------------------------------------------------------------
# Walk-forward evaluation
# --------------------------------------------------------------------------------------
def walk_forward(build, listings, track, cutoffs, horizon=HORIZON_MONTHS, verbose=True,
                 return_model=False):
    """For each cutoff: fit on sales before it, predict sales in the next `horizon` months.
    The homes are passed through data.hide_answers(), exactly like the real test."""
    results = []
    for cut in cutoffs:
        cut = pd.Timestamp(cut)
        end = cut + pd.DateOffset(months=horizon)
        train = listings[listings["date_sold"] < cut]
        test = listings[(listings["date_sold"] >= cut) & (listings["date_sold"] < end)]

        t0 = time.time()
        model = build().fit(train)
        homes = data.hide_answers(test, track)
        res = pd.DataFrame({
            "cutoff": str(cut.date()),
            "month": test["date_sold"].dt.to_period("M").astype(str).to_numpy(),
            "house_type_name": test["house_type_name"].to_numpy(),
            "sold": test["sold_price"].to_numpy(),
        })
        if hasattr(model, "predict_components"):
            comps = model.predict_components(homes)
            for name, p in comps.items():
                res[f"pred_{name}"] = p
            res["pred"] = model.blend(comps)
        else:
            res["pred"] = model.predict(homes)
        results.append(res)
        if verbose:
            print(f"  cutoff {cut.date()}: trained on {len(train):,}, predicted {len(test):,} "
                  f"-> MdAPE {scores(res)['mdape']:.2f}%  ({time.time() - t0:.0f}s)")
    out = pd.concat(results, ignore_index=True)
    return (out, model) if return_model else out


def scores(res, col="pred"):
    """MdAPE (the ranking metric), MAPE (tie-break) and median signed error (bias)."""
    err = 100 * (res[col] - res["sold"]) / res["sold"]
    return {"mdape": float(err.abs().median()), "mape": float(err.abs().mean()),
            "bias": float(err.median())}


def breakdown(res, by="house_type_name"):
    """Error by group, for error analysis (also try by="month" or by="cutoff")."""
    rows = {k: scores(g) | {"n": len(g)} for k, g in res.groupby(by)}
    return pd.DataFrame(rows).T.sort_values("mdape")


# --------------------------------------------------------------------------------------
# Building models from a settings dict
# --------------------------------------------------------------------------------------
def build_from(config, track, learners):
    """{"half_life": 12, "lgbm_params.num_leaves": 63} -> a function returning GBMBlend."""
    kwargs = {"learners": tuple(learners)}
    for key, value in config.items():
        if "." in key:
            group, param = key.split(".", 1)
            kwargs.setdefault(group, {})[param] = value
        else:
            kwargs[key] = value
    make_features = partial(features.gbm_features, track=track)
    return lambda: models.GBMBlend(track, make_features, **kwargs)


def sample_config(rng, learners):
    config = {}
    for key, options in SEARCH_SPACE.items():
        if "." in key and key.split("_params.")[0] not in learners:
            continue   # parameters of a learner that isn't in use
        config[key] = rng.choice(options)
    return config


def best_blend_weight(res, a, b, step=0.05):
    """Weight w on learner a (1 - w on b) that minimises MdAPE on these predictions."""
    best = (None, np.inf)
    for w in np.arange(0, 1 + 1e-9, step):
        blended = np.exp(w * np.log(res[f"pred_{a}"]) + (1 - w) * np.log(res[f"pred_{b}"]))
        mdape = float((100 * (blended - res["sold"]).abs() / res["sold"]).median())
        if mdape < best[1]:
            best = (round(float(w), 2), mdape)
    return best


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["off_market", "list_aware"], required=True)
    ap.add_argument("--learners", default="lgbm", help="comma list of lgbm, catboost, hgb")
    ap.add_argument("--trials", type=int, default=10, help="random settings to try")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", metavar="CONFIG_JSON",
                    help="skip tuning: evaluate this config once on the final fold (Feb-May 2026)")
    ap.add_argument("--ablate-text", action="store_true",
                    help="instead of random settings, compare text=None/flags/tfidf/both")
    args = ap.parse_args()
    learners = args.learners.split(",")
    listings = data.load("listings")

    if args.final:
        with open(args.final) as f:
            config = json.load(f)
        print(f"Final fold, {args.track}, learners={learners}, config={config}")
        res, model = walk_forward(build_from(config, args.track, learners), listings, args.track,
                                  FINAL_CUTOFFS, return_model=True)
        print("\nOverall:", {k: round(v, 2) for k, v in scores(res).items()})
        print("\nBy property type:\n", breakdown(res).round(2))
        print("\nBy month:\n", breakdown(res, "month").round(2))
        if getattr(model, "scorer_", None) is not None:
            print("\nRemarks phrases with the most weight:\n",
                  model.scorer_.top_terms(15).to_string(index=False))
        res.to_csv(f"final_predictions_{args.track}.csv", index=False)
        return

    rng = random.Random(args.seed)
    if args.ablate_text:
        configs = [{"text": t} for t in models.GBMBlend.TEXT_OPTIONS]
    else:
        configs = [{}] + [sample_config(rng, learners) for _ in range(args.trials)]   # {} = defaults
    board = []
    for i, config in enumerate(configs):
        print(f"\nTrial {i}/{len(configs) - 1}: {config or 'defaults'}")
        res = walk_forward(build_from(config, args.track, learners), listings, args.track, TUNE_CUTOFFS)
        s = scores(res)
        row = {"trial": i, **s, "config": json.dumps(config)}
        if len(learners) == 2:
            w, blended = best_blend_weight(res, *learners)
            row.update({f"best_w_{learners[0]}": w, "mdape_best_blend": blended})
        board.append(row)
        print(f"  -> MdAPE {s['mdape']:.2f}%  MAPE {s['mape']:.2f}%  bias {s['bias']:+.2f}%")

    board = pd.DataFrame(board).sort_values("mdape")
    board.to_csv(f"tuning_{args.track}.csv", index=False)
    print("\nTop 5 (tuning folds only):")
    print(board.head(5).to_string(index=False))

    best = json.loads(board.iloc[0]["config"])
    if len(learners) == 2:
        best["weights"] = {learners[0]: float(board.iloc[0][f"best_w_{learners[0]}"]),
                           learners[1]: round(1 - float(board.iloc[0][f"best_w_{learners[0]}"]), 2)}
    path = f"best_{args.track}.json"
    with open(path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nSaved best settings to {path}. When you are done tuning, run once:\n"
          f"  python tune.py --track {args.track} --learners {args.learners} --final {path}")


if __name__ == "__main__":
    main()
