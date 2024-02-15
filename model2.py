from sys import displayhook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import ipywidgets as widgets

hls_all_raw = pd.read_csv("dataset.csv")
print(hls_all_raw)

print(hls_all_raw["Indicator"])
print("\n===========================================================\n")
hls_slice = pd.DataFrame(hls_all_raw, columns =["Country","Indicator","Type of indicator","Time","Value"])
print(hls_slice)

# HI vs LS
# hls_hi = hls_slice.loc[hls_all_raw["Indicator"] == "Household income"]
# hls_ls = hls_slice.loc[hls_all_raw["Indicator"] == "Life satisfaction"]

hls_hi = hls_slice.loc[hls_all_raw["Indicator"] == "Deaths from suicide, alcohol, drugs"]
hls_ls = hls_slice.loc[hls_all_raw["Indicator"] == "Homicides"]

hls_train = hls_hi.loc[hls_hi["Time"] == 2018]
hls_train = hls_train.loc[hls_hi["Type of indicator"] == "Average"]

hls_train1 = hls_ls.loc[hls_ls["Time"] == 2018]
hls_train1 = hls_train1.loc[hls_ls["Type of indicator"] == "Average"]

print("\n===========================================================\n")
print("Record:")
print(hls_train)

print("\n===========================================================\n")
print("Record:")
print(hls_train1)

merged_train_data = pd.merge(hls_train, hls_train1, on="Country")
# merged_train_data = merged_train_data.rename(columns={"Value": "Life satisfaction", "2018": "Household income"})
# merged_train_data = pd.DataFrame(merged_train_data, columns=['Country','Life satisfaction', 'Household income'])

print(merged_train_data)

X = np.c_[merged_train_data["Value_x"]]
Y = np.c_[merged_train_data["Value_y"]]
x = X.tolist()
y = Y.tolist()

# plot data
out1 = widgets.Output()
with out1:
  plt.scatter(x, y)
  plt.xlabel('Percieved health')
  plt.ylabel('Happiness')
  plt.title("Data Plot")
  plt.show()

# fit linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, Y)

# # plot predictions
# predict_x = [x for x in range(910)]
# predict_x = [[x/100] for x in predict_x]
predict_y = model.predict(X)

out2 = widgets.Output()
with out2:
  plt.plot(X, predict_y)
  plt.scatter(x, y)
  plt.xlabel('Percieved health')
  plt.ylabel('Happiness')
  plt.title("Prediction Line")
  plt.show()

displayhook(widgets.HBox([out1,out2]))