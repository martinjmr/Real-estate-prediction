# Real Estate Price Prediction

Predicting the listed price of French properties from listing data, location and photos. Course project in the L3 IASO program (Université Paris Dauphine-PSL), **ranked 1st in the cohort**.

**Best model: tuned XGBoost, 22.9% MAPE against 52.5% for a linear-regression baseline.**

## Data

- Listings: property type, size, rooms, city, GPS coordinates, energy performance, exposure, amenities, and prices for the training set
- Listing photos
- External sources: DVF property transactions (average price per m² by municipality), INSEE median income and population by municipality, points of interest from OpenStreetMap (via OSMnx)

The course data is not included; the notebook expects it in `data/`.

## Approach

1. **Preprocessing**: merging, median imputation (an IterativeImputer based on random forests was also tested, without significant gain), frequency encoding for the city, one-hot encoding for property type and exposure.
2. **Feature engineering**: log of price and size, room density, outdoor-comfort score, total size, RobustScaler.
3. **Location**: K-means with 40 clusters on coordinates and price per m² to define market zones, ranked from cheapest to most expensive; test listings assigned to a zone by nearest neighbours; distance to the nearest point of interest (BallTree, haversine) and to the nearest of France's ten largest cities; DVF and INSEE indicators by municipality.
4. **Photos**: two simple statistics per listing, mean brightness and entropy (OpenCV, scikit-image).
5. **Models**: linear regression as a baseline, random forest, and XGBoost tuned with RandomizedSearchCV in two stages (a wide search, then a narrow one around the best parameters), with K-fold cross-validation.

## Results (cross-validated MAPE)

| Model | MAPE |
|---|---|
| Linear regression | 52.5% |
| Random forest | 29.6% |
| XGBoost | 25.8% |
| XGBoost, tuned | **22.9%** |

## Next steps

- Location embeddings instead of K-means zones
- Image features from a pre-trained CNN instead of brightness and entropy
- Stacking the tested models

## Run it

```bash
pip install -r requirements.txt
jupyter notebook REAL_ESTATE_PREDICTION.ipynb
```

The notebook (in French) runs the full pipeline, from preprocessing to the submission files.
