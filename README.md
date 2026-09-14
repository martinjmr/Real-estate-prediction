Real Estate Price Prediction

Machine learning project predicting the price of real estate listings in France, combining tabular data, geographic enrichment, and image analysis.

Context

A property's price depends on more than its intrinsic characteristics (size, number of rooms, property type). Location, local socio-economic context, and even the visual quality of listing photos play a role. This project explores these different dimensions to build a robust prediction model, combining several heterogeneous data sources.

Data
X_train / X_test: listing features (property type, size, number of rooms, city, GPS coordinates, energy performance, exposure, amenities, etc.)
y_train: prices for the training set
Images: photos associated with each listing
External data: DVF files (French property transaction records), median income per municipality (INSEE), municipal population, points of interest from OpenStreetMap (via OSMnx)
Approach
1. Tabular data preprocessing

Merging datasets, handling missing values (median for numerical variables, "unknown" category for exposure), correlation analysis, dropping uninformative variables (upper_floors, postal_code), encoding categorical variables (frequency encoding for city, one-hot encoding for property_type and exposition).

2. Feature engineering

Log transformation of price and size to correct their skewed distributions, creation of cross features (room density, outdoor comfort score, total size), normalization via RobustScaler.

3. Geographic enrichment
K-means clustering (40 clusters) combining location and price per m² to identify homogeneous market zones, with clusters ranked from cheapest to most expensive (geo_price_rank)
Cluster assignment for the test set via a KNN model based on geographic proximity
Distance to the nearest point of interest (schools, shops, transit...) computed via BallTree and the Haversine distance, using OpenStreetMap data
Distance to the nearest of France's 10 largest cities
Socio-economic enrichment: average price per m² by municipality (DVF), median income by municipality (INSEE), municipal population
4. Image analysis

Extraction of average brightness and entropy from each listing's photos (via OpenCV / scikit-image), followed by correlation analysis with price.

5. Modeling

Three model families tested and compared:

Linear regression (baseline)
Random Forest
XGBoost, with hyperparameter tuning via RandomizedSearchCV (K-Fold cross-validation, two-stage search: broad exploration followed by fine-tuning around the best parameters)

An advanced imputation method (IterativeImputer, based on a RandomForestRegressor) was also tested for handling missing values, without significant improvement over median imputation.

Results
Model	MAPE
Linear regression	52.5%
Random Forest	29.6%
XGBoost (before tuning)	25.8%
XGBoost (after tuning)	22.9%

The tuned XGBoost model is the best model obtained, with a MAPE of 22.9%. The most important features according to the feature importance analysis are shown in feature_importance_xgboost.png.

Tech stack

pandas · numpy · scikit-learn · xgboost · geopandas · osmnx · shapely · geopy · opencv-python · scikit-image · matplotlib · seaborn

Project structure
.
├── REAL_ESTATE_PREDICTION.ipynb   # Main notebook (full pipeline)
├── data/                          # Datasets (not versioned)
└── README.md
Usage
bash
pip install -r requirements.txt
jupyter notebook REAL_ESTATE_PREDICTION.ipynb
Possible improvements
Test other geographic encoding methods (location embeddings)
Use a richer vision model for image features than brightness/entropy alone (e.g. a pretrained CNN)
Stacking or blending of the tested models
