# Next-diagnosis prediction on a synthetic cohort

**Headline.** A 1.07M-parameter causal transformer ranks the next recorded diagnosis better than a tuned age and sex baseline. Mean per-disease AUROC is 0.702 against 0.648. The difference, 0.054, has a 95 percent interval of 0.051 to 0.057 over validation patients, and three training seeds give a mean AUROC of 0.700 to 0.702. It is higher on 454 of 511 diseases and in every sex and age stratum. On 12 of 16 pre-declared cardiometabolic codes the interval for the difference excludes zero. Shuffling the history removes all of its NLL advantage and about a third of its AUROC advantage, so part of the gain depends on the order of the history. The leakage test passes with a positive control.

## 1. Framing

I predicted the next recorded diagnosis token. I chose it because every model, from a frequency table to a transformer, can be scored on it by exactly the same rule, so the comparison is fair. It says what comes next, not when.

At prediction point k the model sees the tokens and ages at positions 0 to k-1. The age of the target is never an input, because it is information from the future. Padding and the two sex tokens are masked at the output. I left out time-to-event prediction, survival heads and ensembling.

I count the result as good if the transformer beats the strongest tuned baseline on held-out NLL and per-disease AUROC, holds that in every sex and age stratum, passes a leakage test with a positive control, and shows that part of the gain depends on the order of the history.

AUROC is computed per disease over all validation prediction points. It measures how well the model ranks the next recorded diagnosis, not patient-level risk.

## 2. Data, checks and assumptions

The cohort is synthetic and uses the Delphi-2M format (Shmatko et al., Nature 2025), so every finding describes the generator. Train has 7,142 patients after I dropped one, and val has 7,143, which gives 172,280 prediction points. The dropped patient, 402867, has no sex token, which breaks the rule that every patient starts with one.

65 tokens appear in val but never in train, on 96 val events. I kept them in the NLL because removing them would flatter every model, and excluded them from AUROC because a disease never seen in training says nothing about ranking. A disease is eligible for AUROC only if it is the target at least 20 times in val. That leaves 511 diseases.

Same-age ties are rare, 243 of 172,280 val prediction points, and show no consistent token order.

## 3. Models

The transformer is causal, with 4 layers, 4 heads and width 128, 1,065,088 parameters in total. It has no positional embedding. A sinusoidal encoding of age is added to each token, because gaps between events are irregular, and a position index would treat a gap of one day and one of twenty years as the same step. Training stopped early after 19 epochs, with the best dev NLL at epoch 15.

The three baselines each isolate one source of signal. Marginal uses overall frequency, age_sex uses sex and a five-year age band, and bigram uses the previous event. The last two are blended with the marginal through a constant K, tuned on a dev split, which chose 1000 for both. The baselines were then refit on all of train, while the transformer trained on 90 percent, so the comparison favours the baselines.

## 4. Results

**Table 1. Validation metrics, same protocol for every model.** Source: `outputs/model_comparison.csv`.

| Model | NLL | Top-1 | Top-5 | Top-20 | Mean AUROC | Median AUROC |
|---|---|---|---|---|---|---|
| marginal | 5.570 | 0.024 | 0.100 | 0.269 | 0.500 | 0.500 |
| age_sex | 5.419 | 0.030 | 0.117 | 0.307 | 0.648 | 0.619 |
| bigram | 5.467 | 0.037 | 0.126 | 0.308 | 0.607 | 0.593 |
| transformer | 5.251 | 0.054 | 0.166 | 0.361 | 0.702 | 0.685 |

The transformer's mean AUROC is 0.054 above age_sex, with a 95 percent interval of 0.051 to 0.057 from 1,000 resamples of val patients. Its NLL is 0.168 nats lower, with an interval of 0.164 to 0.173. Three training seeds give a mean AUROC of 0.700 to 0.702, so the seed spread is small next to the gap.

The transformer has the higher AUROC on 454 of 511 diseases. I give no interval for that count, because the percentile bootstrap is biased low for a count of wins and its upper bound falls below the observed value. Mean AUROC is also higher in every stratum: by 0.062 for female patients, 0.063 for male, 0.050 under age 40, 0.095 from 40 to 60, and 0.059 over 60.

## 5. Is the gain real?

I held the history fixed, replaced every later token and age with random values, and compared the logits at the last history position. Over 150 cases the largest difference was 0.0. A test that always passes proves nothing, so I ran a positive control. For one patient the difference at position k-1 was 0.00 and at position k it was 2.08.

I then shuffled the diagnoses within each val patient and left ages and sex in place. The age_sex baseline, scored on the same shuffle, is the control, because its predictions do not change and only its targets move.

**Table 2. Shuffle control.** Source: `outputs/analysis_summary.json`. Advantage is positive when the transformer is better.

| | NLL | NLL shuffled | AUROC | AUROC shuffled |
|---|---|---|---|---|
| transformer | 5.251 | 5.676 | 0.702 | 0.582 |
| age_sex | 5.419 | 5.645 | 0.648 | 0.547 |
| transformer advantage | 0.168 | -0.031 | 0.054 | 0.035 |

Shuffling removes all of the NLL advantage and about a third of the AUROC advantage. The rest survives, since a shuffled history still holds the patient's own diagnoses. The size is uncertain, because shuffled sequences are out of distribution for the transformer.

The transformer's top-20 lead over age_sex grows from 0.039 at 1 to 4 events seen to 0.077 at 30 or more. That is confounded with age. The Spearman correlation between history length and age is 0.712.

## 6. Cardiometabolic endpoints

I declared a panel before running anything: I20 to I25, I48, I50, I60 to I69 and E11. Table 3 shows five of the codes.

**Table 3. Panel codes, with 95 percent intervals over val patients.** Source: `outputs/bootstrap_panel.csv`.

| Code | Positives | Transformer AUROC | age_sex AUROC | Difference | Interval |
|---|---|---|---|---|---|
| I25 chronic ischaemic heart disease | 1,724 | 0.843 | 0.685 | 0.158 | 0.146 to 0.169 |
| I20 angina pectoris | 1,039 | 0.755 | 0.624 | 0.130 | 0.113 to 0.147 |
| I50 heart failure | 508 | 0.831 | 0.740 | 0.092 | 0.073 to 0.110 |
| E11 non-insulin-dependent diabetes | 1,234 | 0.728 | 0.617 | 0.111 | 0.094 to 0.129 |
| I48 atrial fibrillation and flutter | 1,420 | 0.706 | 0.670 | 0.036 | 0.025 to 0.047 |

Of the 16 eligible codes, 12 have an interval that excludes zero, all in the transformer's favour. The four that include zero are I60, I61, I62 and I69. Three codes, I23, I66 and I68, have too few val targets to be eligible. The intervals are per code and are not corrected for multiple comparisons.

## 7. Data and what the model learnt

Trained on a quarter of the training patients, the transformer reaches a mean AUROC of 0.651, which matches the Table 1 age_sex baseline fitted on all of train, at 0.648. The curve has not flattened, at 0.681 with half the patients and 0.702 with all, though each fraction is a single run.

Delphi-2M reported that its diagnosis embeddings group by ICD-10 chapter, and I tested that on my token embeddings, with the tests fixed before running. The grouping is detectable but very small. An untrained model scores about half as much, which is within what shuffled labels give by chance. The declared check did not come out: H36 is not among the near neighbours of E11, and G63 has too few training events to test.

I show one patient, chosen by a fixed rule. The true next diagnosis ranked 50 under the transformer and 193 under age_sex, and was in neither top 5.

## 8. Calibration and failures

Per disease, I compared observed positives with the total probability assigned. The median ratio is 1.014, with an interquartile range of 0.930 to 1.108, so the total mass per disease is about right. In the highest bin, predicted probability 0.1 and above, the mean prediction is 0.141 and the observed rate is 0.107, over 21,124 pairs. The model is overconfident when it is most confident.

I did not test causes for the 23 failure codes. I measured whether age alone, or the previous event alone, carries signal for each. Age distance is how far the AUROC of age alone sits from one half. 20 of 23 failure codes have no previous-event pair seen at least 5 times in train, against 29.4 percent of all 511 eligible diseases. All 15 low-AUROC codes have a below-median age distance. 5 of the 8 underperforms_age_sex codes have an above-median age distance. The failures are consistent with missing signal of the kinds I measured, not with a fault specific to the model. Lift covers only the previous event, not the whole history.

## 9. Limitations

The intervals hold the trained model fixed, so they cover which patients are in val and not training randomness. Three seeds give a spread, not an interval. Each training fraction is one run. The panel intervals are not corrected for multiple comparisons. K was not retuned per fraction. The data are synthetic, so the findings describe the generator. History length is confounded with age. Shuffled sequences are out of distribution for the transformer.

## 10. Next, not done

None of this is done. I would add a time-to-event head, try temperature scaling for the top bin, test the output head and deeper layers for chapter structure, and add dated biomarker or omics events to the sequence.

Full detail is in `results.ipynb`.
