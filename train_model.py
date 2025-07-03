import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data/advertize.csv")
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

model = LinearRegression()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained by santoshn and saved to model1 and deploy to github.pkl")