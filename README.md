# EV Market & Charging Intelligence

An end-to-end data analytics mini project exploring electric vehicle (EV) specifications and global EV charging infrastructure using **Python, Power BI, and Tableau**.

## 1. Project Overview

The project cleans, analyzes, and visualizes EV and charging-station data to investigate vehicle performance, battery characteristics, driving range, charging capabilities, and the geographic distribution of charging infrastructure.

### Objectives

- Clean and standardize both datasets with Python and Pandas.
- Explore EV manufacturers, battery capacity, driving range, efficiency, and charging power.
- Analyze charging sites, charging ports, power categories, fast-DC availability, and country-level distribution.
- Engineer additional analytical features.
- Build machine-learning models to predict EV driving range.
- Present findings through Python visualizations and interactive Power BI and Tableau dashboards.

## 2. Technology Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine learning:** Scikit-learn
- **Business intelligence:** Microsoft Power BI, Tableau
- **Notebook environment:** Jupyter Notebook

## 3. Datasets

The project uses two public datasets.

### A. Electric Vehicle Specifications Dataset 2025

The report describes 478 EV records and 22 fields. Fields include manufacturer, model, battery capacity, energy efficiency, driving range, fast-charging power, top speed, torque, acceleration, towing capacity, cargo volume, seats, dimensions, drivetrain, segment, and body type.

### B. Global EV Charging Stations 2025

The report describes 242,417 charging sites across 121 countries, with 11 fields. These include station ID and name, city, state/province, country code, latitude, longitude, number of charging ports, maximum charging power, power class, and fast-DC availability.

> **Important:** These datasets describe different entities and are not merged row by row. Analyze them separately and compare their insights conceptually through Power BI and Tableau.

### Suggested project layout

Adapt the filenames below to match the files you actually download.

```text
EV-Market-Charging-Intelligence/
├── data/
│   ├── raw/
│   │   ├── ev_specifications_2025.csv
│   │   └── global_ev_charging_stations_2025.csv
│   └── processed/
├── notebooks/
│   └── ev_market_charging_analysis.ipynb
├── outputs/
│   ├── figures/
│   └── summaries/
├── power-bi/
│   └── ev_market_intelligence.pbix
├── tableau/
│   └── ev_charging_infrastructure_explorer.twbx
├── requirements.txt
└── README.md
```

## 4. Setup and Installation

### Requirements

Install Python 3.10 or later and Jupyter Notebook/JupyterLab. Power BI Desktop and Tableau are separate applications for dashboard development.

### Create a virtual environment

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `requirements.txt` with:

```text
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
jupyter
ipykernel
```

If the `py` launcher is unavailable on Windows, install Python and ensure it is added to PATH. If PowerShell blocks environment activation, use the appropriate execution-policy guidance for your machine or run the environment's Python executable directly.

## 5. Implementation Workflow

### Step 1 — Load the datasets

Place the downloaded CSV files in `data/raw/`. Update the paths in the notebook to match their actual filenames.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ev = pd.read_csv("data/raw/ev_specifications_2025.csv")
charging = pd.read_csv("data/raw/global_ev_charging_stations_2025.csv")

print("EV dataset:", ev.shape)
print("Charging dataset:", charging.shape)

display(ev.head())
display(charging.head())
```

### Step 2 — Inspect and clean the data

Inspect dimensions, data types, missing values, and duplicates before modifying the data. Replace the example column names below with the exact names in your source CSV files.

```python
for name, df in [("EV", ev), ("Charging", charging)]:
    print(f"\n{name} dataset")
    print(df.shape)
    print(df.dtypes)
    print(df.isnull().sum().sort_values(ascending=False).head(15))
    print("Duplicate rows:", df.duplicated().sum())
```

Typical cleaning tasks:

1. Standardize text by trimming unnecessary whitespace.
2. Convert numeric fields to numeric data types.
3. Remove duplicate EV records and duplicate charging station IDs.
4. Investigate missing values and invalid physical measurements.
5. Treat invalid negative physical measurements as missing rather than assuming they are valid.
6. Standardize country codes and fast-DC indicators.
7. Validate charging station latitude and longitude.
8. Check the row counts and data quality after cleaning.

**Example patterns (adjust column names to the source files):**

```python
# Trim whitespace in text columns.
for df in (ev, charging):
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].str.strip()

# Remove duplicate EV rows.
ev = ev.drop_duplicates()

# Remove duplicate stations only if a station ID column exists.
# Replace "station_id" with the actual ID column name.
# charging = charging.drop_duplicates(subset=["station_id"])

# Convert selected columns after checking their actual names.
# ev["battery_capacity"] = pd.to_numeric(
#     ev["battery_capacity"], errors="coerce"
# )
```

Do not copy the example column names blindly. Confirm the actual schema with `ev.columns` and `charging.columns` first.

### Step 3 — Feature engineering

The report proposes these derived indicators. Check units and source column names before applying the formulas.

| Feature | Formula | Purpose |
|---|---|---|
| Range per kWh | Driving range / battery capacity | Compare range relative to battery size |
| Charging power per kWh | Fast-charging power / battery capacity | Compare charging power relative to battery size |
| Estimated kWh per 100 km | Battery capacity / driving range × 100 | Estimate energy use from rated capacity and range |
| Range efficiency index | Driving range / energy efficiency | Additional comparative indicator |
| Power per port | Charging power / number of ports | Compare station power with port count |

Use safeguards to avoid division by zero:

```python
# Example only: substitute your dataset's actual column names.
# ev["range_per_kwh"] = (
#     ev["driving_range"] /
#     ev["battery_capacity"].replace(0, np.nan)
# )
```

The report groups EVs into **Short, Medium, Long, and Very Long** range categories, and charging stations into **Very Low, AC Standard, Fast, High Power, Ultra Fast, and Extreme** power categories. Define and document explicit thresholds based on the dataset's units before creating these categories; the report does not specify numeric cutoffs.

### Step 4 — Exploratory data analysis

Explore and visualize:

**EV analysis**
- Number of EV models by manufacturer.
- Battery-capacity and driving-range distributions.
- Battery capacity versus driving range.
- Energy efficiency versus driving range.
- Battery capacity versus fast-charging power.
- Average range by body type.
- EV models with the longest driving ranges.
- Correlations between major technical specifications.

**Charging infrastructure analysis**
- Charging sites and charging ports by country.
- Distribution of maximum charging power.
- Charging stations by power category.
- Fast-DC versus other charging stations.
- Fast-DC share by country.
- Geographic distribution using validated latitude and longitude.

Example plot (update the column name to match your CSV):

```python
# Example only; use the actual driving-range column.
# sns.histplot(data=ev, x="driving_range", kde=True)
# plt.title("Distribution of EV Driving Range")
# plt.xlabel("Driving Range")
# plt.tight_layout()
# plt.show()
```

### Step 5 — Predict EV driving range

The report specifies **Linear Regression** and **Random Forest** models, evaluated using **MAE, RMSE, and R²**.

Implementation outline:

1. Select suitable EV specification features and the driving-range target.
2. Exclude target leakage and identifiers that do not represent useful predictive information.
3. Handle missing values and categorical features with a preprocessing pipeline.
4. Split the data into training and test sets.
5. Train both models using the same split and comparable preprocessing.
6. Evaluate MAE, RMSE, and R² on held-out data.
7. Plot actual versus predicted range and compare the metrics.

Starter example — replace the placeholder column names with the actual schema and refine preprocessing for your data:

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set these to columns that really exist in your dataset.
target = "driving_range"
features = [
    "battery_capacity",
    "energy_efficiency",
    "fast_charging_power",
]

# This starter uses numeric features only.
# Check the schema and remove rows with a missing target.
model_data = ev[features + [target]].copy()
for col in features + [target]:
    model_data[col] = pd.to_numeric(model_data[col], errors="coerce")
model_data = model_data.dropna(subset=[target])

X = model_data[features]
y = model_data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42
    ),
}

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", estimator),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")
```

**Modeling note:** This starter is for a numeric-only baseline. If you add categorical features, use a `ColumnTransformer` with `OneHotEncoder` for categorical columns. Review the source data for missing or inconsistent units, and avoid using features that directly encode the target. Report actual test metrics rather than assuming one model performs better.

### Step 6 — Export cleaned datasets and summaries

After validating the cleaning and feature-engineering steps, export separate files for downstream dashboard development.

```python
from pathlib import Path

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

ev.to_csv(output_dir / "ev_specifications_cleaned.csv", index=False)
charging.to_csv(
    output_dir / "charging_stations_cleaned.csv", index=False
)
```

You may also export summary tables, such as models by manufacturer, average range by body type, stations by country, and ports by power category. Ensure that all exported tables have descriptive column names and documented units.

## 6. Power BI Dashboard

Import the cleaned EV and charging CSV files separately into Power BI Desktop.

### EV Market Intelligence
Suggested dashboard elements, consistent with the report:
- KPI cards: EV model count, manufacturer count, average range, average battery capacity, and average fast-charging power.
- Bar chart: EV models by manufacturer.
- Scatter plots: battery capacity versus range; efficiency versus range.
- Bar chart: average range by body type.
- Slicers: manufacturer, vehicle type/body type, drivetrain, and range category.

### Global EV Charging Infrastructure
- KPI cards: charging-site count, total ports, average charging power, and fast-DC share.
- Bar charts: charging sites and ports by country.
- Distribution by charging power category.
- Fast-DC versus other stations.
- Map using validated latitude and longitude.
- Slicers: country, power category, and fast-DC availability.

Create measures using the actual imported table and column names. Check whether the dataset uses one row per station and whether port counts are additive before defining totals.

## 7. Tableau Dashboard

Connect Tableau to the cleaned CSV files and build an interactive **EV & Charging Infrastructure Explorer**.

Recommended views:
- Geographic map of charging stations.
- Charging infrastructure by country.
- Charging ports by power class.
- Fast-DC share by country.
- EV manufacturer and driving-range comparisons.
- Battery capacity, energy efficiency, and charging-power relationships.
- KPI cards and interactive filters.

As in Power BI, keep the EV and charging datasets separate unless a valid shared analytical key is available. A side-by-side or dashboard-level comparison does not require a row-level join.

## 8. Run and Verify Checklist

- [ ] Both raw CSV files are in `data/raw/`.
- [ ] Column names and units have been checked.
- [ ] Missing values, duplicates, and invalid measurements have been reviewed.
- [ ] Latitude and longitude values are valid for mapped stations.
- [ ] Derived features use the correct units and avoid division by zero.
- [ ] EV and charging datasets are analyzed separately.
- [ ] Python visualizations run successfully.
- [ ] Linear Regression and Random Forest are evaluated on held-out data.
- [ ] MAE, RMSE, and R² are recorded from actual model runs.
- [ ] Cleaned CSVs are exported to `data/processed/`.
- [ ] Power BI dashboard filters and KPIs work.
- [ ] Tableau map, charts, and filters work.
- [ ] Dashboard totals are checked against the cleaned datasets.

## 9. Expected Outcome

The completed project should turn raw EV specifications and charging infrastructure data into cleaned analytical datasets, Python visualizations, machine-learning evaluation results, and interactive Power BI and Tableau dashboards. It is intended to help users explore EV manufacturers, vehicle performance, battery characteristics, charging capabilities, and charging-network distribution.

The counts and results in the report describe the stated project datasets. Your final README, notebook, and dashboards should reflect the actual files and results produced when you run the implementation.

## 10. Limitations and Reproducibility

- The source report does not provide download URLs, exact CSV filenames, all column names, or numeric thresholds for the derived categories. Fill these in after inspecting the downloaded source files.
- The feature formulas are analytical indicators; verify their units and interpretation before presenting them as physical efficiency measurements.
- The two datasets describe different entities and should not be directly joined without a defensible key.
- Dashboard figures and model metrics can vary with dataset versions, cleaning decisions, selected features, and train/test split.

## License and Data Attribution

Add the original dataset names, publishers, download links, access dates, and license/usage terms here after confirming them from the dataset sources. Do not assume a license or attribution requirement without checking the original source.
