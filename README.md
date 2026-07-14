# Turbo.az — Used Car Price Prediction

An end-to-end regression project on **50,800 used-car listings scraped from [turbo.az](https://turbo.az)**, Azerbaijan's largest automotive marketplace. The goal is to predict a listing's price in AZN from its specifications, and — just as importantly — to establish *which* attributes actually drive price in the Azerbaijani used-car market.

This repository documents the full pipeline: raw-data cleaning, multi-currency normalisation, high-cardinality categorical handling, feature engineering, model training, and feature-importance analysis.

> **Status: work in progress.** The Random Forest baseline is complete. Gradient boosting, cross-validation, and a full model comparison are in progress — see [Roadmap](#roadmap).

---

## Overview

Used-car pricing is a classic tabular regression problem, but the turbo.az data brings three complications that make it a genuine engineering exercise rather than a `.fit()` call:

**1. Prices are quoted in three currencies.** Sellers list in AZN, USD, or EUR. Any model trained on the raw `Price` column would be learning the seller's currency choice as much as the car's value. All prices are normalised to AZN before anything else happens.

**2. The `Model` column has ~1,700 levels.** This is a long-tail categorical: the top 50 models cover 58% of listings, the top 200 cover 83%, and the remaining ~1,500 models share the last 17% — most with fewer than five listings each. No model can learn a reliable price estimate from three examples.

**3. Optional equipment arrives as a delimited string.** The `Supply` column packs a car's entire feature set into one field:

```
Yüngül lehimli disklər;ABS;Lyuk;Yağış sensoru;Kondisioner;Dəri salon;...
```

Treated as a category, this produces thousands of unique combinations with a count of one. It is not a category — it is a **set**, and it has to be encoded as one.

---

## Data

| | |
|---|---|
| **Source** | turbo.az listings (scraped) |
| **Raw size** | 50,800 rows |
| **Target** | `Price_AZN` — listing price normalised to Azerbaijani manat |
| **Format** | Parquet (`turbo_df.parquet`) |

**Feature columns used**

| Feature | Type | Notes |
|---|---|---|
| `Year` | numeric | Manufacture year — the dominant predictor |
| `Engine` | numeric | Displacement |
| `Model`, `Band` | categorical | Nested: model is determined by band (make) |
| `Ban type`, `Color`, `Fuel type`, `Box`, `Gear`, `Condition` | categorical | Low-cardinality |
| `Seats count`, `Owners count` | numeric | Arrive as strings — see cleaning below |
| `Is new`, `Credit`, `Barter` | binary | Yes/No → 1/0 |
| `Supply` | multi-label | Semicolon-delimited equipment set |

Dropped: `Link`, `Classified Date`, `Description`, `Currency`, `Price`, `rate` — either identifiers, free text, or intermediate columns that would leak the target.

---

## Preprocessing

### Currency normalisation

Prices are converted to a single unit before modelling:

```python
rates = {"AZN": 1.0, "USD": 1.70, "EUR": 1.95}
df["rate"] = df["Currency"].map(rates)
df["Price_AZN"] = df["Price"] * df["rate"]
```

Both `Price` and `Currency` are then dropped — retaining either would leak the target back into the feature matrix.

### Type coercion

Two numeric columns arrive as strings because of open-ended top categories:

- `Seats count` contains `"8+"` → mapped to `8`
- `Owners count` contains `"4 və daha çox"` ("4 or more") → mapped to `4`

Both are then passed through `pd.to_numeric(errors="coerce")`, which converts any remaining unparseable value to `NaN` rather than raising — making failures visible instead of silent. Residual missing values are filled with the column median; categorical columns (`Condition`, `Supply`) with the mode.

### Long-tail categorical handling

The `Model` column is truncated to its **200 most frequent levels**, retaining ~83% of rows. This guarantees every surviving model has enough listings (~25+) to support a price estimate, at the cost of discarding the exotic tail. This is a deliberate trade-off — see [Limitations](#limitations).

### Multi-label encoding

`Supply` is expanded into a multi-hot matrix — one binary column per distinct equipment token — rather than being treated as a categorical:

```python
supply_encoded = df["Supply"].str.get_dummies(sep=";").add_prefix("Supply_")
```

A car with 13 features gets 13 ones. This collapses thousands of junk combinations into ~13 clean, high-signal binary columns, each of which the model can actually split on.

### Also applied

- Invalid `Year == 1904` rows dropped (data-entry artefacts)
- Duplicate rows removed
- Remaining categoricals one-hot encoded with `drop_first=True`

---

## Results

**Random Forest** — 300 estimators, default depth, 80/20 split.

| Metric | Train | Test |
|---|---|---|
| R² | *TBD* | *TBD* |
| RMSE (AZN) | *TBD* | *TBD* |

*(Fill these in from your notebook output before pushing.)*

The train-vs-test gap is reported deliberately: a Random Forest with unbounded depth will fit the training set nearly perfectly, and the size of that gap is the honest measure of how much of the score is memorisation.

### Feature importance

Because one-hot encoding shatters a single conceptual feature across dozens of columns (`Model` becomes 199 separate binary features), raw `feature_importances_` is misleading — the importance of "model" is scattered and each fragment looks trivial. Importances are therefore **regrouped back to their source feature** before plotting:

```python
def original_feature_name(col):
    prefixes = ["Model_", "Band_", "Ban type_", "Color_",
                "Fuel type_", "Box_", "Gear_", "Condition_", "Supply_"]
    for prefix in prefixes:
        if col.startswith(prefix):
            return prefix.rstrip("_")
    return col
```

Each group's importances are summed, giving a readable ranking of *which attributes matter*, rather than *which one-hot column matters*.

![Grouped feature importance](figures/feature_importance.png)

---

## Repository Structure

```
.
├── turbo_price_prediction.ipynb   # Main notebook: cleaning → features → model → importance
├── turbo_df.parquet               # Raw scraped listings
├── figures/
│   └── feature_importance.png     # Grouped RF feature importance
├── requirements.txt
└── README.md
```

---

## Requirements

```
pip install pandas numpy scikit-learn matplotlib pyarrow
```

Tested with Python 3.10+.

---

## How to Reproduce

1. Place `turbo_df.parquet` in the repository root.
2. Open `turbo_price_prediction.ipynb` in Jupyter / VS Code / Colab.
3. Run all cells top-to-bottom. Each section prints its own shapes and diagnostics.

---

## Limitations

Stated plainly, because a result without its caveats is not a result:

- **No mileage feature.** Mileage is the second-strongest predictor of used-car price after year, and it is absent from this dataset. The ceiling on achievable accuracy is correspondingly lower than a production pricing model would reach.
- **Exchange rates are hardcoded.** `USD = 1.70`, `EUR = 1.95` are approximations, not the CBAR rate on each listing's date. Since the manat is effectively pegged to the dollar, the USD error is small; the EUR error is not.
- **The exotic tail is discarded.** Filtering to the top 200 models removes ~17% of rows — and those rows are *not* a random sample. They are disproportionately the rare, expensive, high-variance listings. Reported error is therefore representative of the mainstream Baku market, not of the market as a whole.
- **Single train/test split.** One 80/20 split produces one number with real variance in it. Fold-to-fold spread is not yet quantified.
- **Target is not log-transformed.** Car prices are heavily right-skewed; RMSE on the raw AZN scale is dominated by the most expensive listings.
- **One-hot encoding is suboptimal for tree models.** A tree splitting on a sparse binary column can only ask *"is it a Camry?"*, never *"is it one of {Camry, Corolla, Prius}?"* — so groupings cost it enormous depth. Native categorical support (LightGBM / CatBoost) handles this in a single split.

---

## Roadmap

- [x] Data cleaning and currency normalisation
- [x] Multi-label `Supply` encoding
- [x] Random Forest baseline + grouped feature importance
- [ ] **Permutation importance** — sklearn's default impurity-based importance is [biased toward high-cardinality features](https://scikit-learn.org/stable/modules/permutation_importance.html), and `Model` has 200 levels. Permutation importance measures actual predictive contribution instead of split count.
- [ ] Log-transform the target (`log1p` → predict → `expm1`)
- [ ] Linear / Ridge baseline — the honest floor every other model must beat
- [ ] XGBoost with early stopping
- [ ] CatBoost / LightGBM with native categorical handling (no one-hot)
- [ ] MLP baseline for comparison
- [ ] K-fold cross-validation — replace the single split with a mean ± std
- [ ] Final model comparison table: encoding × scaling × CV RMSE × fit time
- [ ] Error analysis — inspect the 10 worst predictions and ask what they have in common

---

## License

MIT