# Real Estate Price Prediction

With my team, I predicted the listed price of French properties from listing data, location and photos. Team project for a course of the IASO bachelor's programme (Université Paris Dauphine-PSL, May 2025), on a Challenge Data competition.

**On the private leaderboard, which gives the challenge's final ranking, our best submission reaches a mean absolute percentage error (MAPE) of 28.1%, against 35.6% for the organisers' XGBoost benchmark: 34th of 219 entries.**

## Data

- Listings: property type, size, rooms, city, GPS coordinates, energy performance, exposure, amenities, and prices for the training set
- Listing photos
- External sources: DVF property transactions (average price per m² by municipality), INSEE population and a 2013 file of median standard of living by municipality, points of interest from OpenStreetMap (via OSMnx)

The challenge data is not included; the notebook expects it in `data/`.

## Approach

1. **Preprocessing**: merging, median imputation (and, in a second version, an IterativeImputer based on random forests), frequency encoding for the city, one-hot encoding for property type and exposure.
2. **Feature engineering**: log of price and size, room density, outdoor-comfort score, RobustScaler.
3. **Location**: K-means with 40 clusters on coordinates and price per m² to define market zones, ranked from cheapest to most expensive; test listings assigned to a zone by nearest neighbours; distance to the nearest of France's ten largest cities (haversine); DVF and INSEE indicators by municipality.
4. **Photos**: two simple statistics per listing, mean brightness and entropy (OpenCV, scikit-image).
5. **Models**: we compared linear regression as a baseline, random forest, and XGBoost tuned with successive RandomizedSearchCV rounds, each narrower than the last, with K-fold cross-validation.

## Results

**Challenge leaderboard** ([Challenge Data](https://challengedata.ens.fr/challenges/68), Institut Louis Bachelier). The final ranking is computed on the private part of the test set.

| Test set | Our MAPE | Benchmark MAPE | Rank |
|---|---|---|---|
| Private (final) | 28.1% | 35.6% | 34th of 219 |

**Validation during the project.** The models predict log(1 + price). The table gives the mean absolute error (MAE) on that scale, as printed in the notebook. An MAE of 0.239 means that the prediction and the true price differ by a factor of about e^0.239 ≈ 1.27, as a geometric mean over the listings.

| Model | Validation | MAE on log(1 + price) |
|---|---|---|
| Linear regression | 30% hold-out | 0.525 |
| Random forest | 20% hold-out | 0.296 |
| XGBoost, first random search | 10-fold cross-validation | 0.258 |
| XGBoost, narrower searches | 5-fold cross-validation | 0.244 |
| XGBoost, narrower searches, after IterativeImputer | 5-fold cross-validation | **0.239** |

Section 12 of the notebook tunes XGBoost on MAPE, but computes it on log(1 + price) instead of prices, and its outputs were not saved; the leaderboard above gives the MAPE on prices.

## Limits

- **The validation errors are optimistic.** The market zones come from K-means on location and price per m², so the zone of a training listing partly reflects its own price, while test listings get theirs from their location only. The leaderboard scores do not have this bias.
- **The distances to points of interest are wrong.** The points of interest were loaded for Île-de-France only, and the saved distances are about 5,400 km for every listing. These columns carry almost no information.

## Next steps

- Location embeddings instead of K-means zones
- Image features from a pre-trained CNN instead of brightness and entropy
- Stacking the tested models

## Run it

```bash
pip install -r requirements.txt
jupyter notebook REAL_ESTATE_PREDICTION.ipynb
```

The notebook (in French) holds the pipeline as it was run during the project, with its saved outputs. Besides the challenge files, it expects in `data/` the DVF files (`DVF/ValeursFoncieres-2020.txt` to `2022`), `donnees_communes.csv`, the 2013 standard-of-living file and the `reduced_images/` folder. Some steps load intermediate CSV files saved during the project (`geo_features3.csv`, `brillance_images.csv`, `entropie_images.csv`) instead of recomputing them.
