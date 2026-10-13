"""
Assignment 1 — Gradio dashboard for the appointment no-show models.

Loads only what train.py saved in artifacts/ (the three trained models, the
held-out test set and summary metrics); nothing is retrained here. Tabs:

- Model comparison: test metrics, agreement between implementations,
  predicted-vs-actual (calibration), loss curves, coefficients.
- Distributions: feature and target distributions by outcome and split.
- Threshold & cost: move the decision threshold and watch the confusion matrix
  and the cost of the reminder-call policy change.
"""

import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from common import (ARTIFACTS, CALL_COST, CALL_EFFECT, MISSED_SLOT_COST,
                    build_standard_model, business_cost)

MODEL_NAMES = ["scikit-learn", "PyTorch manual", "PyTorch nn/optim"]
# First three slots of the reference categorical palette (validated as a set);
# the dashes are a second encoding, since the three models' curves overlap almost exactly.
MODEL_STYLE = {
    "scikit-learn": dict(color="#2a78d6", dash="solid"),
    "PyTorch manual": dict(color="#eb6834", dash="dash"),
    "PyTorch nn/optim": dict(color="#1baf7a", dash="dot"),
}
OUTCOME_COLORS = {"Attended": "#2a78d6", "No-show": "#eb6834"}
LAYOUT = dict(template="plotly_white", height=420, margin=dict(l=60, r=20, t=60, b=50),
              legend=dict(orientation="h", y=-0.2), hovermode="closest")


# ---------------------------------------------------------------- load artifacts

def load_artifacts():
    required = ["sklearn_logreg.joblib", "manual_torch.pt", "standard_torch.pt",
                "test_set.npz", "metrics.json", "appointments.csv"]
    missing = [name for name in required if not (ARTIFACTS / name).exists()]
    if missing:
        raise SystemExit(f"Missing {missing} in {ARTIFACTS}/. "
                         "Run `uv run python main.py train` first.")

    with open(ARTIFACTS / "metrics.json") as f:
        metrics = json.load(f)
    test = np.load(ARTIFACTS / "test_set.npz")
    X, y = test["X"], test["y"]
    X_t = torch.tensor(X)

    sk_model = joblib.load(ARTIFACTS / "sklearn_logreg.joblib")
    manual = torch.load(ARTIFACTS / "manual_torch.pt")
    std_model = build_standard_model(X.shape[1])
    std_model.load_state_dict(torch.load(ARTIFACTS / "standard_torch.pt"))
    std_model.eval()

    with torch.no_grad():
        probs = {
            "scikit-learn": sk_model.predict_proba(X)[:, 1],
            "PyTorch manual": torch.sigmoid(X_t @ manual["w"] + manual["b"]).numpy(),
            "PyTorch nn/optim": std_model(X_t).squeeze(-1).numpy(),
        }
    appointments = pd.read_csv(ARTIFACTS / "appointments.csv")
    return metrics, y, probs, appointments


METRICS, Y_TEST, PROBS, APPTS = load_artifacts()
APPTS["Outcome"] = np.where(APPTS["NoShow"] == 1, "No-show", "Attended")


# ---------------------------------------------------------------- tab 1: comparison

def metrics_table():
    rows = []
    for name, m in METRICS["test_results"].items():
        rows.append({"Model": name, "ROC AUC": m["roc_auc"], "Avg precision": m["avg_precision"],
                     "Log loss": m["log_loss"], "Threshold": m["threshold"],
                     "Calls %": m["calls_pct"], "Precision": m["precision"], "Recall": m["recall"],
                     "Cost / 1000 appts": m["cost_per_1000"]})
    return pd.DataFrame(rows).round(4)


def agreement_table():
    rows = [{"Compared with scikit-learn": name,
             "Max |prob diff|": a["max_abs_prob_diff"], "Mean |prob diff|": a["mean_abs_prob_diff"],
             "Max |weight diff|": a["max_abs_weight_diff"]}
            for name, a in METRICS["agreement"].items()]
    return pd.DataFrame(rows).map(lambda v: f"{v:.2e}" if isinstance(v, float) else v)


def calibration_plot():
    """Predicted vs actual: mean predicted probability vs observed no-show rate per decile."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(color="#9a9a96", width=1, dash="dash"), hoverinfo="skip"))
    for name in MODEL_NAMES:
        prob = PROBS[name]
        bins = pd.qcut(prob, 10, labels=False, duplicates="drop")
        grouped = pd.DataFrame({"p": prob, "y": Y_TEST, "bin": bins}).groupby("bin")
        mean_p, rate, n = grouped["p"].mean(), grouped["y"].mean(), grouped.size()
        fig.add_trace(go.Scatter(
            x=mean_p, y=rate, mode="lines+markers", name=name,
            line=dict(width=2, **MODEL_STYLE[name]), marker=dict(size=9),
            customdata=n, hovertemplate="predicted %{x:.3f}<br>actual %{y:.3f}<br>n=%{customdata}"))
    fig.update_layout(title="Predicted vs actual no-show rate (test set, by predicted-risk decile)",
                      xaxis_title="Mean predicted probability", yaxis_title="Observed no-show rate",
                      **LAYOUT)
    return fig


def coefficient_plot():
    names = METRICS["feature_names"]
    weights = {k: np.array(v) for k, v in METRICS["weights"].items()}
    order = np.argsort(-np.abs(weights["scikit-learn"]))[::-1]
    fig = go.Figure()
    for name in MODEL_NAMES:
        fig.add_trace(go.Bar(y=[names[i] for i in order], x=weights[name][order], name=name,
                             orientation="h", marker_color=MODEL_STYLE[name]["color"],
                             hovertemplate="%{y}: %{x:.4f}"))
    fig.update_layout(title="Coefficients per implementation (standardised features; + = more likely no-show)",
                      xaxis_title="Coefficient (log-odds per 1 SD)", barmode="group",
                      bargap=0.25, bargroupgap=0.1, **{**LAYOUT, "height": 700})
    return fig


def loss_plot():
    fig = go.Figure()
    for name, hist in METRICS["loss_history"].items():
        fig.add_trace(go.Scatter(y=hist, mode="lines", name=name,
                                 line=dict(width=2, **MODEL_STYLE[name])))
    fig.update_layout(title="Training loss (BCE + L2) per epoch, PyTorch implementations",
                      xaxis_title="Epoch", yaxis_title="Loss", yaxis_type="log", **LAYOUT)
    return fig


# ---------------------------------------------------------------- tab 2: distributions

# (bin edges, readable labels) for each numeric column
NUMERIC_BINS = {
    "LeadDays": ([-0.5, 0.5, 2.5, 7.5, 14.5, 30.5, 60.5, 200],
                 ["same day", "1-2", "3-7", "8-14", "15-30", "31-60", "61+"]),
    "Age": ([-1, 12, 18, 30, 45, 60, 75, 101],
            ["0-12", "13-18", "19-30", "31-45", "46-60", "61-75", "76+"]),
    "PriorNoShowRate": ([-0.01, 0.0, 0.25, 0.5, 0.75, 1.0],
                        ["0 / none", "0-25%", "25-50%", "50-75%", "75-100%"]),
}
CATEGORICAL = ["Weekday", "Gender", "SMS_received", "Scholarship", "Hypertension",
               "Diabetes", "Alcoholism", "Handicap", "Neighbourhood"]
DIST_COLUMNS = list(NUMERIC_BINS) + CATEGORICAL


def binned(column, data):
    if column in NUMERIC_BINS:
        edges, labels = NUMERIC_BINS[column]
        return pd.cut(data[column], edges, labels=labels).astype(str)
    if column == "Neighbourhood":   # 81 levels: show the 15 largest
        top = data[column].value_counts().index[:15]
        return data[column].where(data[column].isin(top), "Other")
    return data[column].astype(str)


def distribution_plots(column, split):
    data = APPTS if split == "all" else APPTS[APPTS["Split"] == split]
    groups = binned(column, data)
    present = set(groups.unique())
    order = ([label for label in NUMERIC_BINS[column][1] if label in present]
             if column in NUMERIC_BINS else groups.value_counts().index.tolist())
    if column == "Weekday":
        order = [d for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                 if d in order]

    counts = pd.crosstab(groups, data["Outcome"]).reindex(order).fillna(0)
    fig_counts = go.Figure()
    for outcome in ["Attended", "No-show"]:
        fig_counts.add_trace(go.Bar(x=counts.index, y=counts.get(outcome, 0), name=outcome,
                                    marker_color=OUTCOME_COLORS[outcome],
                                    marker_line=dict(width=2, color="white"),
                                    hovertemplate="%{x}<br>" + outcome + ": %{y}"))
    fig_counts.update_layout(title=f"Appointments by {column} ({split})", barmode="stack",
                             xaxis_title=column, yaxis_title="Appointments", **LAYOUT)

    rate = data.groupby(groups)["NoShow"].agg(["mean", "size"]).reindex(order)
    overall = data["NoShow"].mean()
    fig_rate = go.Figure(go.Bar(x=rate.index, y=rate["mean"], marker_color="#eb6834",
                                customdata=rate["size"], name="No-show rate",
                                hovertemplate="%{x}<br>no-show rate %{y:.1%}<br>n=%{customdata}"))
    fig_rate.add_hline(y=overall, line=dict(color="#52514e", width=1, dash="dash"),
                       annotation_text=f"overall {overall:.1%}")
    fig_rate.update_layout(title=f"No-show rate by {column} ({split})", xaxis_title=column,
                           yaxis_title="No-show rate", yaxis_tickformat=".0%", **LAYOUT)
    return fig_counts, fig_rate


# ---------------------------------------------------------------- tab 3: threshold & cost

THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def threshold_view(model, threshold, call_cost, missed_cost, effect):
    prob = PROBS[model]
    call = prob >= threshold
    tn = int((~call & (Y_TEST == 0)).sum())
    fp = int((call & (Y_TEST == 0)).sum())
    fn = int((~call & (Y_TEST == 1)).sum())
    tp = int((call & (Y_TEST == 1)).sum())
    costs = dict(call_cost=call_cost, missed_slot_cost=missed_cost, call_effect=effect)

    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]], x=["Not called", "Called"], y=["Attended", "No-show"],
        text=[[f"TN<br>{tn:,}", f"FP<br>{fp:,}"], [f"FN<br>{fn:,}", f"TP<br>{tp:,}"]],
        texttemplate="%{text}", textfont=dict(size=16), showscale=False, xgap=2, ygap=2,
        colorscale=[[0, "#cde2fb"], [1, "#1c5cab"]],
        hovertemplate="%{y} / %{x}: %{z:,}<extra></extra>"))
    cm.update_layout(title=f"Confusion matrix — {model}, threshold {threshold:.2f} (test set)",
                     xaxis_title="Decision", yaxis_title="Actual", yaxis_autorange="reversed",
                     **{**LAYOUT, "height": 380})

    curve = np.array([business_cost(Y_TEST, prob >= t, **costs) for t in THRESHOLD_GRID])
    per_1000 = curve / len(Y_TEST) * 1000
    current = business_cost(Y_TEST, call, **costs)
    nobody = business_cost(Y_TEST, np.zeros(len(Y_TEST)), **costs)
    everyone = business_cost(Y_TEST, np.ones(len(Y_TEST)), **costs)

    fig_cost = go.Figure(go.Scatter(x=THRESHOLD_GRID, y=per_1000, mode="lines", name="Model policy",
                                    line=dict(width=2, color="#2a78d6"),
                                    hovertemplate="threshold %{x:.2f}<br>cost %{y:,.0f} per 1000"))
    fig_cost.add_hline(y=nobody / len(Y_TEST) * 1000, line=dict(color="#52514e", dash="dash", width=1),
                       annotation_text="call nobody")
    fig_cost.add_vline(x=threshold, line=dict(color="#eb6834", width=2),
                       annotation_text=f"current {threshold:.2f}")
    fig_cost.update_layout(title="Expected cost per 1,000 appointments vs threshold (test set)",
                           xaxis_title="Threshold", yaxis_title="Cost per 1,000 appointments",
                           showlegend=False, **{**LAYOUT, "height": 380})

    breakeven = call_cost / (effect * missed_cost) if effect * missed_cost > 0 else float("nan")
    n = len(Y_TEST)
    summary = f"""
**{tp + fp:,} calls** ({(tp + fp) / n:.1%} of {n:,} test appointments) ·
**{tp:,} of {tp + fn:,} no-shows reached** (recall {tp / max(tp + fn, 1):.1%}, precision {tp / max(tp + fp, 1):.1%})

| Policy | Total cost | Per 1,000 appts |
|---|---|---|
| **This threshold ({threshold:.2f})** | **{current:,.0f}** | **{current / n * 1000:,.0f}** |
| Call nobody | {nobody:,.0f} | {nobody / n * 1000:,.0f} |
| Call everyone | {everyone:,.0f} | {everyone / n * 1000:,.0f} |

Saving vs calling nobody: **{nobody - current:,.0f}** ({(nobody - current) / nobody:.1%}).
Threshold chosen on validation (default costs): **{METRICS['thresholds'][model]:.2f}**.
Break-even probability for a call, call cost / (effect × missed-slot cost): {breakeven:.2f}.
"""
    return cm, fig_cost, summary


# ---------------------------------------------------------------- layout

hp = METRICS["hyperparameters"]
with gr.Blocks(title="Appointment no-show dashboard") as demo:
    gr.Markdown(f"""
# Appointment no-show prediction
The clinic phones patients flagged as likely no-shows. Logistic regression trained three ways
(scikit-learn, a manual PyTorch loop, `nn.Module` + `torch.optim`) on appointments up to
May 20, 2016, tuned on May 24–31, evaluated on **June 1–8** ({len(Y_TEST):,} appointments,
{Y_TEST.mean():.1%} no-shows). Chosen hyperparameters: C = {hp['C']} (L2 = {hp['l2']:.1e}),
manual learning rate {hp['manual_lr']}, Adam learning rate {hp['standard_lr']}.
All models are loaded from `artifacts/`; nothing is retrained here.
""")

    with gr.Tab("Model comparison"):
        gr.Markdown("### Test-set results (three implementations vs naive baseline)")
        gr.Dataframe(metrics_table(), interactive=False)
        gr.Markdown("### Do the three implementations agree?")
        gr.Dataframe(agreement_table(), interactive=False)
        with gr.Row():
            gr.Plot(calibration_plot())
            gr.Plot(loss_plot())
        gr.Plot(coefficient_plot())

    with gr.Tab("Distributions"):
        with gr.Row():
            column = gr.Dropdown(DIST_COLUMNS, value="LeadDays", label="Feature")
            split = gr.Radio(["all", "train", "val", "test"], value="all", label="Split")
        with gr.Row():
            counts_plot = gr.Plot()
            rate_plot = gr.Plot()
        for control in (column, split):
            control.change(distribution_plots, [column, split], [counts_plot, rate_plot])
        demo.load(distribution_plots, [column, split], [counts_plot, rate_plot])

    with gr.Tab("Threshold & cost"):
        with gr.Row():
            model = gr.Dropdown(MODEL_NAMES, value="scikit-learn", label="Model")
            threshold = gr.Slider(0.01, 0.99, value=METRICS["thresholds"]["scikit-learn"],
                                  step=0.01, label="Decision threshold (call if P(no-show) ≥ threshold)")
        with gr.Row():
            call_cost = gr.Number(CALL_COST, label="Cost of one reminder call")
            missed_cost = gr.Number(MISSED_SLOT_COST, label="Cost of a missed slot")
            effect = gr.Slider(0, 1, value=CALL_EFFECT, step=0.05,
                               label="Share of no-shows a call prevents")
        with gr.Row():
            cm_plot = gr.Plot()
            cost_plot = gr.Plot()
        summary = gr.Markdown()
        inputs = [model, threshold, call_cost, missed_cost, effect]
        outputs = [cm_plot, cost_plot, summary]
        for control in inputs:
            control.change(threshold_view, inputs, outputs)
        model.change(lambda m: METRICS["thresholds"][m], model, threshold)
        demo.load(threshold_view, inputs, outputs)


if __name__ == "__main__":
    demo.launch()
