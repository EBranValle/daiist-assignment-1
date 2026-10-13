"""
Assignment 1 — training pipeline: predicting medical appointment no-shows.

Business framing (see REPORT.md): the clinic phones patients the model flags
as likely no-shows the day before their appointment. The model is trained on
past appointments and used on future ones, so the split is by appointment
date, every feature uses only information available before the appointment,
and the decision threshold is chosen to minimise the cost defined in
common.business_cost, not to maximise accuracy.

Steps:
1. Load and clean data/noshows.csv.
2. Engineer features (lead time, patient history from earlier dates only, ...).
3. Split by AppointmentDay into train / validation / test.
4. Train logistic regression three ways on the same standardised features:
   scikit-learn, a manual PyTorch loop, and nn.Module + torch.optim.
5. Pick the regularisation strength and each model's threshold on the
   validation set; report everything on the test set against a naive baseline.
6. Save models and everything app.py needs to artifacts/.
"""

import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import ARTIFACTS, DATA_PATH, build_standard_model, business_cost

SEED = 0
VAL_START = pd.Timestamp("2016-05-24")   # train: Apr 29 – May 20
TEST_START = pd.Timestamp("2016-06-01")  # val: May 24 – May 31, test: Jun 1 – Jun 8

AGE_BINS = [-1, 12, 18, 30, 45, 60, 75, 200]
AGE_LABELS = ["0-12", "13-18", "19-30", "31-45", "46-60", "61-75", "76+"]
AGE_REFERENCE = "31-45"            # dropped one-hot level (absorbed by the bias)
MIN_NEIGHBOURHOOD_COUNT = 100      # rarer neighbourhoods (in train) -> "OTHER"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]   # sklearn inverse regularisation strength
# Learning rates picked once by trying a few values: both converge well within
# these epochs (see the loss curves in the dashboard).
MANUAL_LR, MANUAL_EPOCHS = 1.0, 3000
STANDARD_LR, STANDARD_EPOCHS = 0.03, 1500


# ---------------------------------------------------------------- data

def load_and_clean():
    df = pd.read_csv(DATA_PATH, parse_dates=["ScheduledDay", "AppointmentDay"])
    df = df.rename(columns={"Hipertension": "Hypertension", "Handcap": "Handicap",
                            "Date.diff": "LeadDays"})
    n_raw = len(df)

    # Booked after the appointment date: impossible, so a data error (5 rows).
    df = df[df["LeadDays"] >= 0].copy()
    # A handful of ages above 100 (max 115) look like entry errors; cap them.
    df["Age"] = df["Age"].clip(upper=100)
    # Positive class = the event the clinic acts on.
    df["NoShow"] = (~df["Showed_up"]).astype(int)

    print(f"Loaded {n_raw} rows, dropped {n_raw - len(df)} with negative lead time.")
    return df.sort_values(["AppointmentDay", "AppointmentID"]).reset_index(drop=True)


def add_patient_history(df):
    """Per appointment: how often this patient booked / no-showed on EARLIER days.

    Grouping by (patient, day) first means appointments on the same day never
    see each other's outcome, which would not be known yet when we call.
    """
    daily = (df.groupby(["PatientId", "AppointmentDay"])
               .agg(day_appts=("NoShow", "size"), day_noshows=("NoShow", "sum"))
               .reset_index()
               .sort_values(["PatientId", "AppointmentDay"]))
    by_patient = daily.groupby("PatientId")
    daily["PriorAppts"] = by_patient["day_appts"].cumsum() - daily["day_appts"]
    daily["PriorNoShows"] = by_patient["day_noshows"].cumsum() - daily["day_noshows"]
    daily = daily.rename(columns={"day_appts": "SameDayAppts"})

    df = df.merge(daily[["PatientId", "AppointmentDay", "SameDayAppts",
                         "PriorAppts", "PriorNoShows"]],
                  on=["PatientId", "AppointmentDay"], how="left")
    df["PriorNoShowRate"] = np.where(df["PriorAppts"] > 0,
                                     df["PriorNoShows"] / df["PriorAppts"].clip(lower=1), 0.0)
    df["AgeBand"] = pd.cut(df["Age"], AGE_BINS, labels=AGE_LABELS).astype(str)
    df["Weekday"] = df["AppointmentDay"].dt.day_name()
    df["Split"] = np.select([df["AppointmentDay"] < VAL_START,
                             df["AppointmentDay"] < TEST_START],
                            ["train", "val"], default="test")
    return df


def build_features(df, neighbourhoods):
    """Model inputs. One-hot groups drop a reference level to avoid collinearity."""
    X = pd.DataFrame(index=df.index)
    # Lead time: strongest signal, very right-skewed -> log, plus same-day flag
    # (same-day bookings almost always show up, a jump a log alone can't make).
    X["log_lead_days"] = np.log1p(df["LeadDays"])
    X["same_day_booking"] = df["LeadDays"] == 0
    # Only interpretable together with lead time (SMS only sent for 3+ day leads).
    X["sms_received"] = df["SMS_received"]

    # Patient history (earlier days only) + a flag for "no history yet".
    X["has_history"] = df["PriorAppts"] > 0
    X["log_prior_appts"] = np.log1p(df["PriorAppts"])
    X["prior_noshow_rate"] = df["PriorNoShowRate"]
    X["multiple_appts_same_day"] = df["SameDayAppts"] > 1

    X["male"] = df["Gender"] == "M"
    for col in ["Scholarship", "Hypertension", "Diabetes", "Alcoholism", "Handicap"]:
        X[col.lower()] = df[col]

    # No-show rate vs age is not monotonic (peaks at 13-30), so bands, not raw age.
    for band in AGE_LABELS:
        if band != AGE_REFERENCE:
            X[f"age_{band}"] = df["AgeBand"] == band
    for day in WEEKDAYS[1:]:                       # Monday = reference
        X[f"weekday_{day}"] = df["Weekday"] == day

    hood = df["Neighbourhood"].where(df["Neighbourhood"].isin(neighbourhoods), "OTHER")
    for name in neighbourhoods[1:]:                # most common one = reference
        X[f"hood_{name}"] = hood == name
    X["hood_OTHER"] = hood == "OTHER"
    return X.astype("float32")


# ---------------------------------------------------------------- models

def bce_loss(y_pred, y):
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    return -(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)).mean()


def train_sklearn(X, y, C):
    return LogisticRegression(C=C, max_iter=10_000, tol=1e-8).fit(X, y)


def train_manual(X, y, l2, lr=MANUAL_LR, n_epochs=MANUAL_EPOCHS):
    """Session 5 style: raw tensors, autograd, hand-written gradient step.

    Loss = mean BCE + l2/2 * ||w||^2, which is scikit-learn's objective divided
    by (C * n), with l2 = 1 / (C * n). The bias is not penalised (same as sklearn).
    """
    torch.manual_seed(SEED)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for epoch in range(n_epochs):
        y_pred = torch.sigmoid(X @ w + b)
        loss = bce_loss(y_pred, y) + l2 / 2 * (w ** 2).sum()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        history.append(loss.item())
    return w.detach(), b.detach(), history


def train_standard(X, y, l2, lr=STANDARD_LR, n_epochs=STANDARD_EPOCHS):
    """Standard workflow: nn.Module + nn.BCELoss + torch.optim.Adam, same objective."""
    torch.manual_seed(SEED)
    model = build_standard_model(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    history = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(X).squeeze(-1)
        loss = loss_fn(y_pred, y) + l2 / 2 * (model[0].weight ** 2).sum()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return model, history


# ---------------------------------------------------------------- evaluation

def best_threshold(y, prob):
    """Threshold that minimises business cost on the given (validation) data."""
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    costs = [business_cost(y, prob >= t) for t in grid]
    return float(grid[int(np.argmin(costs))])


def evaluate(y, prob, threshold):
    call = prob >= threshold
    tp = int((call & (y == 1)).sum())
    return {
        "roc_auc": roc_auc_score(y, prob),
        "avg_precision": average_precision_score(y, prob),
        "log_loss": log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7)),
        "threshold": threshold,
        "calls_pct": float(call.mean() * 100),
        "precision": tp / max(int(call.sum()), 1),
        "recall": tp / max(int((y == 1).sum()), 1),
        "cost_per_1000": business_cost(y, call) / len(y) * 1000,
    }


FEATURE_GROUPS = {
    "lead time": lambda f: f in ("log_lead_days", "same_day_booking"),
    "SMS": lambda f: f == "sms_received",
    "patient history": lambda f: f in ("has_history", "log_prior_appts",
                                       "prior_noshow_rate", "multiple_appts_same_day"),
    "age bands": lambda f: f.startswith("age_"),
    "health/welfare flags": lambda f: f in ("male", "scholarship", "hypertension",
                                            "diabetes", "alcoholism", "handicap"),
    "weekday": lambda f: f.startswith("weekday_"),
    "neighbourhood": lambda f: f.startswith("hood_"),
}

# Dropped from the final model because of the ablation below: leaving either
# out improves validation log loss (the ~60 neighbourhood dummies overfit the
# training weeks; weekday carries almost no signal). Re-check if data changes.
DROPPED_GROUPS = ("neighbourhood", "weekday")


def feature_ablation(X_train, y_train, X_val, y_val, feature_names, C=1.0):
    """Validation AUC / log loss when each candidate feature group is left out (sklearn)."""
    def score(cols):
        prob = train_sklearn(X_train[:, cols], y_train, C).predict_proba(X_val[:, cols])[:, 1]
        return {"val_roc_auc": roc_auc_score(y_val, prob), "val_log_loss": log_loss(y_val, prob)}

    rows = {"all features": score(list(range(len(feature_names))))}
    for group, in_group in FEATURE_GROUPS.items():
        rows[f"without {group}"] = score([i for i, f in enumerate(feature_names) if not in_group(f)])
    print("\nFeature-group ablation (validation set):")
    print(pd.DataFrame(rows).T.round(4).to_string())

    # The SMS confounding: its sign flips when lead time is left out.
    no_lead = [i for i, f in enumerate(feature_names) if not FEATURE_GROUPS["lead time"](f)]
    model = train_sklearn(X_train[:, no_lead], y_train, C)
    sms_coef = model.coef_.ravel()[[feature_names[i] for i in no_lead].index("sms_received")]
    print(f"SMS coefficient without lead-time features: {sms_coef:+.3f}\n")


def main():
    ARTIFACTS.mkdir(exist_ok=True)

    df = add_patient_history(load_and_clean())
    train_mask, val_mask, test_mask = (df["Split"] == s for s in ("train", "val", "test"))
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        part = df[mask]
        print(f"{name:5s}: {len(part):6d} rows, {part['AppointmentDay'].min().date()} to "
              f"{part['AppointmentDay'].max().date()}, no-show rate {part['NoShow'].mean():.3f}")

    # Neighbourhood levels come from the training period only.
    counts = df.loc[train_mask, "Neighbourhood"].value_counts()
    neighbourhoods = counts[counts >= MIN_NEIGHBOURHOOD_COUNT].index.tolist()
    candidates = build_features(df, neighbourhoods)
    y_all = df["NoShow"].to_numpy()
    y_train, y_val, y_test = y_all[train_mask], y_all[val_mask], y_all[test_mask]

    # Which feature groups earn their place? Judged on validation, never test.
    X_cand = StandardScaler().fit(candidates[train_mask]).transform(candidates).astype("float32")
    feature_ablation(X_cand[train_mask], y_train, X_cand[val_mask], y_val,
                     candidates.columns.tolist())

    features = candidates[[c for c in candidates.columns
                           if not any(FEATURE_GROUPS[g](c) for g in DROPPED_GROUPS)]]
    feature_names = features.columns.tolist()
    print(f"{candidates.shape[1]} candidate features, {len(feature_names)} kept "
          f"(dropped: {', '.join(DROPPED_GROUPS)}).")

    scaler = StandardScaler().fit(features[train_mask])     # fit on train only
    X_all = scaler.transform(features).astype("float32")
    X_train, X_val, X_test = X_all[train_mask], X_all[val_mask], X_all[test_mask]

    # 1) scikit-learn: choose regularisation C on validation log loss.
    val_scores = {}
    for C in C_GRID:
        prob = train_sklearn(X_train, y_train, C).predict_proba(X_val)[:, 1]
        val_scores[C] = log_loss(y_val, prob)
        print(f"sklearn C={C:<6}  val log loss {val_scores[C]:.5f}")
    best_C = min(val_scores, key=val_scores.get)
    sk_model = train_sklearn(X_train, y_train, best_C)
    l2 = 1.0 / (best_C * len(y_train))   # the same penalty, rescaled for a mean loss
    print(f"-> C={best_C} (l2={l2:.2e} for the PyTorch versions)")

    # 2) Manual PyTorch loop and 3) standard nn.Module + torch.optim, same objective.
    Xt_train, yt_train = torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float32)
    w, b, manual_hist = train_manual(Xt_train, yt_train, l2)
    std_model, standard_hist = train_standard(Xt_train, yt_train, l2)

    def predict_all(X):
        Xt = torch.tensor(X)
        with torch.no_grad():
            return {"scikit-learn": sk_model.predict_proba(X)[:, 1],
                    "PyTorch manual": torch.sigmoid(Xt @ w + b).numpy(),
                    "PyTorch nn/optim": std_model(Xt).squeeze(-1).numpy()}

    val_prob, test_prob = predict_all(X_val), predict_all(X_test)

    # Naive baseline: predict the training no-show rate for everyone (nobody gets called).
    results = {"Baseline (train rate)":
               evaluate(y_test, np.full(len(y_test), y_train.mean()), 0.5)}
    thresholds = {}
    for name in test_prob:
        thresholds[name] = best_threshold(y_val, val_prob[name])   # chosen on validation
        results[name] = evaluate(y_test, test_prob[name], thresholds[name])
    print("\nTest set results:")
    print(pd.DataFrame(results).T.round(4).to_string())

    # Do the three implementations agree?
    weights = {"scikit-learn": sk_model.coef_.ravel(), "PyTorch manual": w.numpy(),
               "PyTorch nn/optim": std_model[0].weight.detach().numpy().ravel()}
    agreement = {}
    for name in ["PyTorch manual", "PyTorch nn/optim"]:
        diff = np.abs(test_prob[name] - test_prob["scikit-learn"])
        agreement[name] = {"max_abs_prob_diff": float(diff.max()),
                           "mean_abs_prob_diff": float(diff.mean()),
                           "max_abs_weight_diff": float(np.abs(weights[name] - weights["scikit-learn"]).max())}
    print("\nAgreement with scikit-learn on the test set:")
    print(pd.DataFrame(agreement).T.to_string())

    # ------------------------------------------------------------ save
    joblib.dump(sk_model, ARTIFACTS / "sklearn_logreg.joblib")
    torch.save({"w": w, "b": b}, ARTIFACTS / "manual_torch.pt")
    torch.save(std_model.state_dict(), ARTIFACTS / "standard_torch.pt")
    np.savez_compressed(ARTIFACTS / "test_set.npz", X=X_test, y=y_test)

    display_cols = ["Split", "NoShow", "LeadDays", "Age", "Gender", "Weekday",
                    "Neighbourhood", "SMS_received", "Scholarship", "Hypertension",
                    "Diabetes", "Alcoholism", "Handicap", "PriorNoShowRate"]
    df[display_cols].to_csv(ARTIFACTS / "appointments.csv", index=False)

    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump({
            "test_results": results,
            "thresholds": thresholds,
            "agreement": agreement,
            "weights": {k: v.tolist() for k, v in weights.items()},
            "feature_names": feature_names,
            "hyperparameters": {"C": best_C, "l2": l2, "manual_lr": MANUAL_LR,
                                "standard_lr": STANDARD_LR},
            "loss_history": {"PyTorch manual": manual_hist, "PyTorch nn/optim": standard_hist},
        }, f, indent=2)
    print(f"\nSaved models and artifacts to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
