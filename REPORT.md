# Assignment 1 Report

- **Name**: Edgar Andres Bran Valle
- **Student ID**: 18789
- **Email**: ebran.ieu2022@student.ie.edu
- **Group**: BBADBA 5B 
## Dataset

*What is it, where did you get it, what does one row represent, how many
rows/columns, and why did you pick it.*

A dataset about no-show medical appointments found in Kaggle. A row represents a medical appointment with metadata on patient ID, gender, age, neighbourhood, health details, when the appointment was booked, if an SMS reminder was sent, and wether a patient showed up to their appointment. The dataset has a total of 106,982 rows with 15 columns. No particular reason to specifically pick this dateset, I was browsing kaggle looking in the datasets tab and thought this dataset was interesting to use for the assignment. 

## Business / real-life framing

*The hypothetical scenario this model serves, and what that scenario implies
for how you built the pipeline — target definition, whether a time-based
split was necessary and why (or why not), which metric should drive the
decision threshold and why.*

## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

| Implementation | How it works | Settings |
|---|---|---|
| scikit-learn | `LogisticRegression().fit()` finds the weights automatically | C = 10, tuned on validation (all C ≥ 0.1 scored the same) |
| Manual PyTorch loop | Gradient descent written by hand: predict, compute loss, backprop, update, reset | 3,000 epochs, learning rate 1.0 |
| Standard PyTorch | `nn.Linear` + `nn.Sigmoid`, trained with the Adam optimiser | 1,500 epochs, learning rate 0.03 |


| Model | ROC AUC | Avg precision | Log loss | Threshold | Calls | Precision | Recall | Cost / 1,000 appts |
|---|---|---|---|---|---|---|---|---|
| Naive baseline (predict train rate 21.1%) | 0.5000 | 0.1859 | 0.4822 | – | 0.0% | – | 0.0% | 4,646.43 |
| scikit-learn | 0.7285 | 0.3361 | 0.4269 | 0.30 | 29.15% | 32.88% | 51.58% | 4,510.49 |
| PyTorch manual loop | 0.7285 | 0.3361 | 0.4269 | 0.30 | 29.25% | 32.86% | 51.71% | 4,510.61 |
| PyTorch nn.Module + optim | 0.7285 | 0.3361 | 0.4269 | 0.30 | 29.25% | 32.86% | 51.71% | 4,510.61 |

| Compared with scikit-learn | Max abs. probability diff | Mean abs. probability diff | Max abs. weight diff |
|---|---|---|---|
| PyTorch manual loop | 0.000621 | 0.000062 | 0.000407 |
| PyTorch nn.Module + optim | 0.000621 | 0.000062 | 0.000407 |

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
