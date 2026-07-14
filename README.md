# Turbo.az Car Price Prediction

A machine learning project for predicting used-car prices from listings collected from **[turbo.az](https://turbo.az)**. The project covers data cleaning, preprocessing, categorical encoding, Random Forest regression, model evaluation, and grouped feature-importance analysis.

## Project Overview

The goal of this project is to build a regression model that estimates a vehicle's price using characteristics such as:

- Brand and model
- Manufacturing year
- Body type
- Engine volume
- Horsepower
- Fuel type
- Transmission and drivetrain
- Vehicle condition
- Number of seats and owners
- Equipment and comfort features
- Credit and barter availability

The original dataset contains **50,833 listings and 22 columns**.

## Dataset

Source: [`vrashad/turbo_az`](https://huggingface.co/datasets/vrashad/turbo_az) — turbo.az car listings, Parquet, licensed CC BY-NC-4.0.

Main columns include:

```text
Band, Model, Year, Ban type, Color, Engine volume, Horsepower,
Fuel type, Box, Gear, Is new, Seats count, Owners count,
Condition, Supply, Credit, Barter, Price, Currency
```

> The dataset itself may not be included in the repository if it is too large or restricted. Update this section with the dataset source when publishing it.

## Data Preprocessing

The preprocessing workflow includes:

1. Removing an invalid listing with the year `1904`
2. Converting seat and owner counts to numeric values
3. Replacing values such as `8+` and `4 və daha çox`
4. Filling missing numerical values with the median
5. Filling missing categorical values with the mode
6. Converting USD and EUR prices to AZN
7. Removing unused text and metadata columns
8. Removing duplicate rows
9. Keeping the 200 most frequent car models
10. One-hot encoding categorical columns
11. Multi-hot encoding the semicolon-separated `Supply` column
12. Converting Yes/No columns into binary values

After preprocessing, the dataset contains approximately:

```text
42,010 rows
306 features
```

## Model

The project currently uses a:

```text
RandomForestRegressor
```

Main configuration:

```python
RandomForestRegressor(
    n_estimators=300,
    n_jobs=-1
)
```

The data is divided into:

- 80% training data
- 20% testing data
- `random_state=42`

## Model Performance

Current model results:

| Metric | Training Set | Test Set |
|---|---:|---:|
| R² Score | 0.9951 | 0.9667 |
| RMSE | 2,264.68 AZN | 6,208.03 AZN |

The test R² score indicates that the model explains approximately **96.67% of the variance** in car prices on the test set.

> The difference between training and test performance suggests some overfitting, although the model still performs strongly on unseen data.

## Feature Importance

The one-hot encoded features are grouped back into their original categories so that the overall importance of groups such as `Model`, `Band`, `Color`, and `Fuel type` can be interpreted more clearly.

![Grouped feature importance](figures/feature_importance.png)

To save the plot in the correct location, create an `images` folder and use:

```python
import os

os.makedirs("images", exist_ok=True)

plt.tight_layout()
plt.savefig(
    "images/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
```

## Repository Structure

```text
Turbo_az_Car_Price_Prediction/
│
├── data_preprocessing.ipynb
├── requirements.txt
├── README.md
│
└── figures/
    └── feature_importance.png
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Yunis-Allahverdi/Turbo_az_Car_Price_Prediction.git
cd Turbo_az_Car_Price_Prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
pandas
numpy
scikit-learn
matplotlib
pyarrow
jupyter
```

## Usage

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
data_preprocessing.ipynb
```

Run the notebook cells from top to bottom.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Parquet / PyArrow

## Possible Future Improvements

- Compare Random Forest with XGBoost, LightGBM, and CatBoost
- Tune hyperparameters using RandomizedSearchCV
- Apply cross-validation
- Investigate and reduce model overfitting
- Add MAE and MAPE evaluation metrics
- Handle rare brands and models without discarding as many listings
- Build a reusable preprocessing pipeline
- Save the trained model with Joblib
- Create a Streamlit web application for price prediction
- Add SHAP-based model explanations
- Add automated tests and a prediction script

## Author

**Yunis Allahverdi**

- GitHub: [Yunis-Allahverdi](https://github.com/Yunis-Allahverdi)

## License

This project is intended for educational and portfolio purposes. Add a license file before allowing reuse or redistribution.
