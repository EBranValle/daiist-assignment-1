# Assignment 1 Report

- **Name**: Edgar Andrès Bran Valle
- **Student ID**: 18789
- **Email**: ebran.ieu2022@student.ie.edu
- **Group**: BBADBA 5B 
## Dataset

A dataset about no-show medical appointments found in Kaggle. A row represents a medical appointment with metadata on patient ID, gender, age, neighbourhood, health details, when the appointment was booked, if an SMS reminder was sent, and wether a patient showed up to their appointment. The dataset has a total of 106,982 rows with 15 columns. No particular reason to specifically pick this dateset, I was browsing kaggle looking in the datasets tab and thought this dataset was interesting to use for the assignment. 

https://www.kaggle.com/datasets/iamtanmayshukla/healthcare-no-shows-appointments-dataset/data

## Business / real-life framing

This model addresses the scenario of the likely probability that a patient shows up or not to their medical appointment. The idea is that no shows cost the hospital money and waste doctors' time and a slot to attend another patient. This model, in theory, would flag patients as likely no shows and help the medical staff phone the patients to give a reminder to reduce the likelihood of a now show.  The target is NoShow in the case that the column Showedup=FALSE. In this data, a time based split was necessary to train the model. The column for appointment day with the Train being dates between Apr 29 – May 20 (65,139 appointments), Validation May 24 – 31 (16,162 appointments), and Test Jun 1 – 8 (25,681 appointments). Of course, the reason we train with earlier dates and test on later dates in to try and predict future likely no shows, another reason is that patients can appear multiple times in the dataset, and a random split would make the model learn from a patient's later appointments.  In this model, the hypothetical thinking is that calling a patient to remind them of their appointment is cheaper than the patient missing the appointment entirely. For this reason, I set the assumption that calling a patient costs 2 dollars and a patient not showing up costs 25 to make it a real business problem and I also set the assumption that a call can help prevent 30% of no shows.  With this, I set a threshold of calling patients with a 30% no-show rate for their appointments; this 0.30 came from testing the values on the validation split.The reason being that is that a call is worth making more than not because it costs less than what it would potentially save. In the data, 80% of patients already show up, so it wouldn't make sense to create a model that already predicts everything and says that it has an 80% chance of a patient showing up without flagging anyone. 


## Data preparation & feature engineering

So first, as part of the data cleaning process, I started by removing 5 rows where the booking date was later than the appointment date; for the first feature, it didn't make sense as it would create negative values. I also just corrected some spelling errors in the column names; other than that, there were no NAs. 

Features: All the features were standardised using the training data’s mean and standard deviation only. 

1.) Lead Time: how many days between booking and appointment. Most bookings happen between 0-2 days but some went up to 170 days so I applied a Log Transformation to squash the big numbers. Also, a flag was added for same day bookings, as most of these tend to show up to their appointment. 
2.)Patient History: Has a patient skipped before? For each patient how many earlier appointments a person had and the number they skipped and calculated a past no-show rate (skipped ÷ earlier appointments). Only earlier appointments are used and not same day appointments since outcomes aren’t known yet; using them would leak the answer. A “Has history” flag separates new patients from returning patients who have never skipped, since both have a no-show rate of 0 . In the case of the dataset about 40% of patients have made a prior appointment. 
3.)Age Groups: age was split into the following age groups. 0–12, 13–18, 19–30, 31–45, 46–60, 61–75, 76+. No shows vary much with age, the groups with the biggest no show percentage (25%) are the groups between 13-30. Each group is a one-hot (yes/no) column. 

Gender, scholarship, hypertension, diabetes, alcoholism and handicap were included as yes/no features. SMSis used with lead time as most are done only to patients with +3 days of lead time.. Their effects are small. 

Features tried and removed: Neibourhood made validation results worse reducing the AUC as it caused overfitting. Weekday was removed as it added nothing. 


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

The no show rate drifts over time. 
The rate falls from 21% to 19.4% in the validation set and 18.6% in the test. This may make the models overestimate risks. For this reason I would retrain regularly the models the more weeks could come out. 

For this project teh cost assumptions are made up.
The costs applied to the models and the call effectives are assumptions. So with this in mind each model saves only about 136 dollars per 1000 appointments as explained in the results. For a real world scenario I would get the real costs in order to make it a real business model solution. 

The SMS is not casual
In the models the SMS coefficient flips depending on whether lead time is included. Because SMS was sent to only booking made 3 days ahead and not at random. Therefore the results cant provide clear evidence that SMS works. A random test would give an unbiased estimate of the effect. 

Patient history starts from zero.
The daset covers only 6 weeks of time. Early training rows have almost no history while test rows at the end of the window have more. This could have an impact on results and pior cancelation histories. Therefore its important to have a full history of every patient in the dataset.

The models only find only about half the no shows. 
All the models have essentially the same results. The data lacks more field that probably drive a patient to do a no show. Like time of day, distance to clinic, weather and appointment type. This sort of data could help narrow down results to also help it be a better business model solution. 

## Generative AI use disclosure

Generative AI use disclosure

I used Claude (Anthropic’s AI assistant, via Claude Code) in this assignment:

Code: Claude wrote the code in train.py, app.py and common.py: the data cleaning, the feature engineering with suggestions from my part, recommended the time-based split, the three logistic regression implementations (scikit-learn, manual PyTorch loop, nn.Module + torch.optim), the evaluation, and the Gradio dashboard. 

Design choices: Claude proposed the features, the removal of neighbourhood and weekday based on the validation ablation, and the cost-based threshold. 

Understanding: Claude explained the pipeline, the results and the concepts to me (threshold, precision/recall, Simpson’s paradox, why the three implementations agree). I wrote the text of this report myself; Claude checked my drafts for factual accuracy. 

I ran the pipeline and the dashboard myself, and the numbers in this report come from my run and checks.
