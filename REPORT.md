# Assignment 1 Report

- **Name**: Edgar Andres Bran Valle
- **Student ID**: 18789
- **Email**: ebran.ieu2022@student.ie.edu
- **Group**: BBADBA 5B 
## Dataset

A dataset about no-show medical appointments found in Kaggle. A row represents a medical appointment with metadata on patient ID, gender, age, neighbourhood, health details, when the appointment was booked, if an SMS reminder was sent, and wether a patient showed up to their appointment. The dataset has a total of 106,982 rows with 15 columns. No particular reason to specifically pick this dateset, I was browsing kaggle looking in the datasets tab and thought this dataset was interesting to use for the assignment. 

## Business / real-life framing

This model addresses the scenario of the likely probability that a patient shows up or not to their medical appointment. The idea is that no shows cost the hospital money and waste doctors' time and a slot to attend another patient. This model, in theory, would flag patients as likely no shows and help the medical staff phone the patients to give a reminder to reduce the likelihood of a now show.  The target is NoShow in the case that the column Showedup=FALSE. In this data, a time based split was necessary to train the model. The column for appointment day with the Train being dates between Apr 29 – May 20 (65,139 appointments), Validation May 24 – 31 (16,162 appointments), and Test Jun 1 – 8 (25,681 appointments). Of course, the reason we train with earlier dates and test on later dates in to try and predict future likely no shows, another reason is that patients can appear multiple times in the dataset, and a random split would make the model learn from a patient's later appointments.  In this model, the hypothetical thinking is that calling a patient to remind them of their appointment is cheaper than the patient missing the appointment entirely. For this reason, I set the assumption that calling a patient costs 2 dollars and a patient not showing up costs 25 to make it a real business problem and I also set the assumption that a call can help prevent 30% of no shows.  With this, I set a threshold of calling patients with a 30% no-show rate for their appointments; this 0.30 came from testing the values on the validation split.The reason being that is that a call is worth making more than not because it costs less than what it would potentially save. In the data, 80% of patients already show up, so it wouldn't make sense to create a model that already predicts everything and says that it has an 80% chance of a patient showing up without flagging anyone. 


## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

## Modeling: three implementations, one model

Since this is a classification problem and not numerical, I used logistic regression. So in turn, the model will output probabilities from 0 to 1 on how likely a patient is likely to not show up to their appointment. As per the assignment, scikit-learn, a manual PyTorch loop, and the standard torch workflows were used in the data. All with the same features and an L2 penalty to keep the weights small and prevent overfitting. The specifications are in the next table. 

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

As we can see from ROC AUC, average precision and the log loss of the results the models defenetly do learn something when comparing it to the Naive Baseline. At the threshold of 0.3 the models all converge on 29% call sent and reaches 52% of all no shows. 1 out of 3 patients would have still missed the appointment even with a call made (33% precision) The cost falls to about 4,510 per 1,000 appointments, a saving of roughly 136 dollars, or 2.9%, versus doing nothing. 

The three models all agree. The results are virtually identical from one another. The decision metric slightly differs by 0.1%. Scikit-learn calls 29.15% of patients vs 29.25% for the PyTorch models. The Pythorch models match each other exactly. The loss is convex and there is a single optimum, wich is why in the end all three converged. 

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

## Generative AI use disclosure

Used Claude AI to help me set up and create the models.
