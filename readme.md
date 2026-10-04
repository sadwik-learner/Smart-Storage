# SmartStorage

## AI-Powered Inventory Decision Intelligence System

### Aim

SmartStorage is an AI-powered inventory decision-intelligence system designed to help businesses make better inventory and replenishment decisions.

The system aims to analyze historical sales and inventory data, forecast future demand, identify potential stockout and overstock situations, and recommend what to order, when to order, and how much to order.

The project combines machine learning, inventory optimization, and Agentic AI to support explainable and human-approved replenishment decisions.



### Repository Structure

```text
minor project/
│
├── dataset/
│   └── retail_store_inventory.csv
│
├── 01_EDA.ipynb
├── 02_Modeling.ipynb
├── README.md
├── requirements.txt
└── SmartStorage_EDA_Summary.docx
```

### Folder and File Guide

#### `dataset/`

Contains the canonical dataset used by the entire team.

```text
dataset/
└── retail_store_inventory.csv
```

The `retail_store_inventory.csv` file is the agreed canonical dataset for the project. The original dataset should not be modified.

#### `01_EDA.ipynb`

Contains the Exploratory Data Analysis of the retail dataset, including:

- Dataset structure and dimensions
- Data types
- Missing-value analysis
- Duplicate analysis
- Numerical statistics
- Categorical analysis
- Sales analysis
- Inventory analysis
- Time-series analysis
- Correlation analysis
- Important dataset observations and limitations

#### `02_Modeling.ipynb`

Contains the shared modeling notebook scaffold.

The modeling stage will include:

- Data preparation
- Feature engineering
- Leakage-safe feature selection
- Time-aware data splitting
- Model training
- Model evaluation
- Experiment comparison

#### `SmartStorage_EDA_Summary.docx`

Contains the detailed report of the exploratory data analysis and the major findings from the dataset.

#### `requirements.txt`

Contains the Python libraries required to run the project notebooks and models.

### Dataset

The primary dataset used for SmartStorage is the **Retail Store Inventory & Demand Forecasting** dataset.

The dataset contains information related to:

- Stores
- Products
- Categories
- Regions
- Inventory levels
- Units sold
- Units ordered
- Demand forecast
- Price
- Discount
- Weather conditions
- Holiday/promotion
- Competitor pricing
- Seasonality

The dataset contains approximately 73,100 records and 15 features.

### EDA Findings

The initial EDA found:

- No missing values
- No duplicate records
- Five stores and twenty products
- Approximately two years of daily observations
- A moderate relationship between inventory level and units sold
- A very high correlation between `Demand Forecast` and `Units Sold`
- A very high correlation between `Price` and `Competitor Pricing`

The existing `Demand Forecast` feature will not be blindly used in our own forecasting model because its extremely high relationship with actual sales may introduce target leakage.

### Modeling Approach

The modeling stage will focus on:

1. Demand forecasting
2. Stockout-risk prediction
3. Overstock and dead-stock analysis
4. Inventory optimization

The models will be evaluated using time-aware validation to avoid using future information during training.

### Leakage-Safe Feature Policy

The current agreed feature set includes relevant historical and contextual variables such as:

- Store ID
- Product ID
- Category
- Region
- Inventory Level
- Units Sold
- Units Ordered
- Price
- Discount
- Weather Condition
- Holiday/Promotion
- Seasonality

`Demand Forecast` is excluded initially because of its extremely high correlation with actual sales. Its use will only be reconsidered after verifying how the original value was generated.

### Canonical Dataset Rule

`dataset/retail_store_inventory.csv` is the canonical dataset for the project.

All team members must use the same version of this file.

The original dataset should not be manually modified. Data cleaning and feature engineering should be performed through reproducible code in the notebooks.

### Team Contribution

Each team member should commit their own work to the repository.

Team members should work on separate branches and use descriptive commit messages.

Example:

```bash
git checkout -b feature/demand-forecasting

git add .
git commit -m "feat: add demand forecasting baseline"

git push origin feature/demand-forecasting
```

Changes should be reviewed and merged into the main branch through the team's agreed Git workflow.