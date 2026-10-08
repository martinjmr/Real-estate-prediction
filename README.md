# Real Estate Price Prediction

With my team, I predicted the listed price of French properties from listing data, location and photos. Team project for a course of the L3 IASO program (Université Paris Dauphine-PSL), **ranked 1st in the cohort**.

**On the challenge's test set, our best submission reaches a mean absolute percentage error (MAPE) of 27.1%, against 36.8% for the organisers' XGBoost benchmark: 20th of 220 entries on the public leaderboard.**

## Data

- Listings: property type, size, rooms, city, GPS coordinates, energy performance, exposure, amenities, and prices for the training set
- Listing photos
- External sources: DVF property transactions (average price per m² by municipality), INSEE median income and population by municipality, points of interest from OpenStreetMap (via OSMnx)

The course data is not included; the notebook expects it in `data/`.

## Approach

1. **Preprocessing**: merging, median imputation (we also tested an IterativeImputer based on random forests, without significant gain), frequency encoding for the city, one-hot encoding for property type and exposure.
2. **Feature engineering**: log of price and size, room density, outdoor-comfort score, total size, RobustScaler.
3. **Location**: K-means with 40 clusters on coordinates and price per m² to define market zones, ranked from cheapest to most expensive; test listings assigned to a zone by nearest neighbours; distance to the nearest point of interest (BallTree, haversine) and to the nearest of France's ten largest cities; DVF and INSEE indicators by municipality.
4. **Photos**: two simple statistics per listing, mean brightness and entropy (OpenCV, scikit-image).
5. **Models**: we compared linear regression as a baseline, random forest, and XGBoost tuned with RandomizedSearchCV in two stages (a wide search, then a narrow one around the best parameters), with K-fold cross-validation.

## Results

**Challenge leaderboard** ([Challenge Data](https://challengedata.ens.fr/challenges/68), Institut Louis Bachelier). The test set is split into a public part, scored after each submission, and a private part, which gives the final ranking.

| Test set | Our MAPE | Benchmark MAPE | Rank |
|---|---|---|---|
| Public | 27.1% | 36.8% | 20th of 220 |
| Private (final) | 28.1% | 35.6% | 34th of 219 |

**Validation during the project.** The models predict log(1 + price). The table gives the mean absolute error (MAE) on that scale, as printed in the notebook. An MAE of 0.239 means that the prediction and the true price differ by a factor of about e^0.239 ≈ 1.27, as a geometric mean over the listings.

| Model | Validation | MAE on log(1 + price) |
|---|---|---|
| Linear regression | 30% hold-out | 0.525 |
| Random forest | 20% hold-out | 0.296 |
| XGBoost, first random search | 5-fold cross-validation | 0.257 |
| XGBoost, refined random search | 5-fold cross-validation | **0.239** |

Section 10 of the notebook computes the MAPE on prices for the validation split, but its output was not saved; the leaderboard above gives the MAPE on the test set.

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
