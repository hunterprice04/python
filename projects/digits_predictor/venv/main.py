import digits_predictor as digits_predictor
from sklearn import datasets

dp = digits_predictor.DigitsPredictor(datasets.load_digits())
dp.display_data()

# this model does not fit the data well at all
# dp.kmeans_model(.25,40,show=True)

# this model fits and predicts the data with 99.1 percent accuracy
dp.svc_model(.25,40)
