# Next-diagnosis prediction on a synthetic health-trajectory cohort

## What is this?

This is my solution to an interview task. I built a small model that reads a patient's history of diagnoses and predicts which diagnosis is recorded next. The cohort is synthetic and uses the data format of Delphi-2M (Shmatko et al., Nature 2025), where each patient is a sequence of tokens and each token carries an age. I took the format from that paper and nothing else. No Delphi code is used here. The model, the baselines and the evaluation are written from scratch in PyTorch and NumPy.

## What did I predict, and why that target?

I predicted the next recorded diagnosis token. At prediction point k the model sees the tokens and ages at positions 0 to k-1 and nothing after. The age of the target event is never an input, because knowing when the next event happens is information from the future. At the output I mask the padding index and the two sex tokens, so no probability lands on a token that can never be a target.

I chose this target because it is the simplest one the data format supports, and because every model, from a frequency table to a transformer, can be scored on it by exactly the same rule. That made the comparison fair before I trained anything. I left out time-to-event prediction, survival heads and ensembling. Each of those would have added a second thing to get right, and I wanted one question answered well: does a sequence model beat strong count-based baselines on held-out patients, and does the gain come from the history?

## What do you need to run it?

You need Python 3.11.7 and the packages pinned in `requirements.txt`. Start from a fresh environment, so nothing already installed can change the versions.

```bash
conda create -n health-trajectories python=3.11.7 -y
conda activate health-trajectories
pip install -r requirements.txt
```

I developed and ran everything on macOS. The code asks for Apple MPS first and falls back to CPU if MPS is missing or fails a small test, which is what `select_device` in `src/model.py` does. The analysis step recorded its device as `mps` in `outputs/analysis_summary.json`. The leakage test inside that step always runs on CPU, and the same file records that too. Training and evaluation ran on the same machine through the same device selection, but those two scripts only print their device and do not save it, so I do not claim more than that.

## Where does the data go?

The data comes from the link in the task PDF. Put the three files in a folder called `data/` at the repo root.

```
data/train.bin
data/val.bin
data/labels.csv
```

The folder is gitignored, so the data is never committed. Each `.bin` file is a flat array of little-endian unsigned 32-bit integers in triples: person id, age in days, token. The token is stored as the vocabulary index minus one, and `load_split` in `src/data.py` adds the one back. `labels.csv` is the vocabulary, one name per line, where the line number is the index and index 0 is padding.

## How do you run it, and in what order?

Run every command from the repo root. The order matters, because later steps read files that earlier steps write.

1. Check the data. The first command prints the sanity checks and saves two histograms. The second counts same-age ties and prints only.

   ```bash
   python -m src.data
   python check_ties.py
   ```

2. Fit and score the baselines. This tunes the interpolation constant K on a dev split of train, refits on all of train, and scores on val.

   ```bash
   python -m src.baselines
   ```

3. Train the transformer. This saves the best checkpoint by dev NLL to `checkpoints/best.pt`.

   ```bash
   python -m src.train
   ```

4. Score the transformer on val and build the headline table.

   ```bash
   python -m src.run_eval
   ```

5. Run the seven analyses. Piping through `tee` keeps the printed report as a file next to the other outputs, and `2>&1` also captures error output.

   ```bash
   python -m src.analysis 2>&1 | tee outputs/analysis_log.txt
   ```

6. Open the walkthrough and run all cells.

   ```bash
   python -m notebook results.ipynb
   ```

The `checkpoints/` folder is gitignored, so a fresh clone has no trained model. Step 3 has to run before steps 4 and 5, and before section 8 of the notebook, which loads the checkpoint for a positive control. Step 5 also needs the baseline files from step 2 and the transformer files from step 4.

Training is quick. Summing the `elapsed_sec` column of `outputs/train_log.csv` gives 146.9 seconds over 19 epochs on my machine. The cap is 30 epochs and early stopping waits 4 epochs without improvement, both set in `src/train.py`.

## What does each step produce?

Everything lands in `outputs/`, which is committed so the results can be read without running anything. The notebook reads from this folder and never writes to it. Its last cell checks that.

| File | What it holds |
|---|---|
| `events_per_patient.png` | Step 1. Histogram of events per patient, train and val. |
| `age_at_event.png` | Step 1. Histogram of age at each event, train and val. |
| `baseline_K_tuning.csv` | Step 2. Dev NLL for every K in the grid, for the age_sex and bigram baselines. |
| `baselines_summary.csv` | Step 2. Val metrics for the three baselines. |
| `marginal_per_disease_auc.csv` | Step 2. AUROC and positive count per eligible disease, marginal baseline. |
| `marginal_stratified_auc.csv` | Step 2. The same within each sex and each age band. |
| `age_sex_per_disease_auc.csv` | Step 2. AUROC and positive count per eligible disease, age_sex baseline. |
| `age_sex_stratified_auc.csv` | Step 2. The same within each sex and each age band. |
| `bigram_per_disease_auc.csv` | Step 2. AUROC and positive count per eligible disease, bigram baseline. |
| `bigram_stratified_auc.csv` | Step 2. The same within each sex and each age band. |
| `train_log.csv` | Step 3. Train loss, dev NLL and seconds for each epoch. |
| `train_curve.png` | Step 3. Train loss and dev NLL against epoch. |
| `transformer_per_disease_auc.csv` | Step 4. AUROC and positive count per eligible disease, transformer. |
| `transformer_stratified_auc.csv` | Step 4. The same within each sex and each age band. |
| `model_comparison.csv` | Step 4. The headline table. All four models, same metrics, same protocol. |
| `per_disease_paired.csv` | Step 5. Transformer and age_sex AUROC side by side for each disease, with the difference. |
| `per_disease_scatter.png` | Step 5. That pairing as a scatter, coloured by how common the disease is. |
| `stratified_comparison.csv` | Step 5. Mean AUROC per stratum for transformer and age_sex. |
| `stratified_bar.png` | Step 5. That table as a bar chart. |
| `transformer_shuffled_per_disease_auc.csv` | Step 5. Transformer AUROC per disease after shuffling each patient's diagnosis order. |
| `transformer_shuffled_stratified_auc.csv` | Step 5. The same within each stratum. |
| `age_sex_shuffled_per_disease_auc.csv` | Step 5. The age_sex baseline on the same shuffled sequences, as the control. |
| `age_sex_shuffled_stratified_auc.csv` | Step 5. The same within each stratum. |
| `history_length.csv` | Step 5. NLL and top-20 accuracy by how many events the model has seen, both models. |
| `history_length.png` | Step 5. That table as two line charts. |
| `calibration.csv` | Step 5. Mean predicted probability, observed rate and count for each decade bin of predicted probability. |
| `calibration_per_disease.csv` | Step 5. Observed over expected for each disease. |
| `calibration.png` | Step 5. Observed rate against mean predicted probability on log axes. |
| `failures.csv` | Step 5. The diseases with the lowest transformer AUROC and the ones where it trails age_sex most. |
| `auc_vs_prevalence.png` | Step 5. Transformer AUROC against disease frequency, failures marked. |
| `analysis_summary.json` | Step 5. Every number the printed summary paragraph uses, plus the device, the tuned K and the shuffle seed. |
| `analysis_log.txt` | Step 5. The printed report of the analysis run, captured with `tee`. |

## What did I assume, and why?

I dropped one training patient, 402867. The sanity check in `src/data.py` requires every patient to start with exactly one sex token, and this patient does not. The age_sex baseline has nothing to condition on for such a patient, so I removed them from every training computation instead of patching the record. The id is a named constant in `src/train.py`, `src/baselines.py`, `src/run_eval.py` and `src/analysis.py`, so all four scripts drop the same person.

Some tokens appear in val but never in train. I kept them in the vocabulary and I kept them in the NLL, because a model that is deployed has to spread probability over things it has not seen, and removing them would flatter every model. I excluded them from per-disease AUROC, because ranking performance on a disease the model had no chance to learn says nothing about the model. The count of these tokens and of the events that carry them is computed live in section 2 of `results.ipynb`.

I checked same-age ties with `check_ties.py`. When two events share an age, the file has to store them in some order, and if that order followed the token index the model could learn the storage order instead of anything about health. The script counts how often the target has the same age as the last history event, and whether tied diagnoses lean ascending or descending. Its output is in section 2 of the notebook.

The age of the target is never an input. Passing it would tell the model when the next event happens, which is information from the future. The evaluation in `src/evaluate.py` hands each model the tokens and ages before the target and nothing else.

The transformer trains on 90 percent of the training patients and uses the other 10 percent as a dev set for early stopping. The split is `split_train_dev` in `src/data.py`, with fraction 0.1 and seed 0. The baselines use that same dev set to tune K, then refit on all of train with the tuned K. So the baselines see slightly more data than the transformer does. I accepted that because it makes the comparison harder for the transformer, not easier. The validation split was not used for any training or tuning decision.

AUROC here is pooled next-event AUROC per disease. For one disease, every prediction point in val is a row, the label is whether the next recorded event is that disease, and the score is the probability the model gave it. A disease is eligible if it is the target at least 20 times in val, is not a sex token and was seen in train, as set in `src/evaluate.py`. This is not patient-level risk over a time horizon. It answers whether the model ranks the right moments highest for each disease, and I do not read it as anything more.

## What is unusual here?

The transformer has no positional embedding. Each token embedding is added to a sinusoidal encoding of the patient's age in years at that event, so age is the only signal of where an event sits in time. I did this because the gaps between events are irregular, and a position index would treat a gap of one day and a gap of twenty years as the same step. It also means two events at the same age get the same time signal, so how ties are stored matters.

The file `src/baselines.py` defines a default K of 20. No reported number uses it. Both conditional baselines are scored with a K tuned on the dev split over the grid in `tune_K`, and the tuned value is 1000 for both according to `outputs/baseline_K_tuning.csv`. The value 20 only ever appears as the first point of that grid. I mention it so nobody reads the default and thinks the baselines were left weak.

## How do I know it reproduces?

The seeds are fixed in the code. Training uses seed 42 for PyTorch and NumPy, set in `src/train.py`. The train and dev split uses seed 0, the default in `src/data.py`. The shuffle control uses seed 0, the constant `_SHUFFLE_SEED` in `src/analysis.py`, and the leakage test uses seed 0 as well.

I tested one part directly. After the results were first committed, I ran `python -m src.run_eval` again. The two files it regenerates from the checkpoint, `transformer_per_disease_auc.csv` and `transformer_stratified_auc.csv`, came out byte for byte identical to the versions committed in `a68b5fd`. Git shows no later commit touching either file even though both were rewritten on disk.

I then built a fresh conda environment from `requirements.txt` and ran `python -m src.run_eval` again. `model_comparison.csv` and both transformer CSVs came out identical to the committed versions.

The analysis also checks itself. It recomputes the age_sex baseline's NLL and stops with an error if it differs from `outputs/baselines_summary.csv` by more than the tolerance in `src/analysis.py`. The notebook does the same for the transformer's NLL between `model_comparison.csv` and `analysis_summary.json`.

There is a limit to what this shows. I trained once, with one seed. Re-running evaluation on a fixed checkpoint was deterministic in the one re-run I did, but I have not measured how much the results move between training runs.

The committed notebook was executed in the environment that trained the model: Python 3.11.7, with the package versions in `requirements.txt`.

## Where are the results?

Open `results.ipynb`. It is committed with its outputs, so it reads without running. It walks through the data checks, the headline table, the per-disease and per-stratum comparisons, the shuffle control, the leakage test with its positive control, history length, calibration, the failures and the limitations. Every number in it is read from a file in `outputs/` or computed in the cell that shows it.

For the headline table alone, read `outputs/model_comparison.csv`. For the numbers behind the analysis summary, read `outputs/analysis_summary.json`.
