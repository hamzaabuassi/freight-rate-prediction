# Freight Rate Prediction

Predicts `posted_rate` for the 12,000 loads in `validation.csv` (Nov-Dec 2025) using the labeled loads in `train-test.csv` (Jan-Oct 2025).

## Setup
Python 3.10+ is recommended.
```bash
pip install -r requirements.txt
```

## Data
The input CSVs are not included in this repo. Place them in the same folder as `pipeline.py` (or in a `data/` folder next to it):
- `train-test.csv`
- `validation.csv`
- `validation-predictions-template.csv`
- `december-chart-inputs.csv`

## Run
```bash
python pipeline.py              # data checks + time-based CV + final model + predictions + December chart
python pipeline.py --skip-cv    # same, without the cross-validation (faster)
```
Results are written to `outputs/`:
- `validation_predictions.csv`: submission file (`load_id,predicted_rate`)
- `december_chart_predictions.csv`: completed December chart inputs
- `cv_results.csv`, `data_quality.json`, `quote_signal_by_month.csv`, `feature_importance.csv`

## Validate the outputs with the provided scorer
```bash
python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_chart_predictions.csv
```
This checks the file formats and creates the December chart in `scorer_results/`.

## Approach
- **Split / validation:** expanding-window, out-of-time CV (train on months before m, test on month m, m = Jun-Oct), since the real test period is in the future.
- **Cleaning:** negative weights converted to absolute value, missing weight imputed by equipment median, missing `market_index` imputed by same-day median, corrupted training labels (rate per mile about 4.5x or 0.2x the norm) removed from training only.
- **Features:** distance, haversine distance, equipment, weight (+flags), `market_index`, daily `market_index` / `quote_signal`, day of week, coordinates. City names are not used, so unseen cities still work.
- **Model:** LightGBM on `log(posted_rate)` with L1 loss. Final prediction is a blend of two model families (with and without `quote_signal`), 3 seeds each.