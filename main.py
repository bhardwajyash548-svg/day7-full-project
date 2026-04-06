import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# load data
df = pd.read_csv("students_project.csv")

# DATA CLEANING 
df["Sleep"].fillna(df["Sleep"].mean(),  inplace = True)

df["Marks"].fillna(df["Marks"].mean(), inplace = True)

df["Hours"].fillna(df["Hours"].mean(), inplace=True)

# FEAURES
# Fill all NaNs in the features with their respective column means
x = df[["Hours", "Sleep"]].fillna(df[["Hours", "Sleep"]].mean())
y = df["Marks"].fillna(df["Marks"].mean())



# model
model = LinearRegression()
model.fit(x, y)

# user input limit 
while True:
    Hours = float(input("enter study Hours (0 to exit:)"))
    if Hours == 0:
        break

    Sleep = float(input(" enter sleep hours:"))

    prediction = model.predict([[Hours, Sleep]])

    print(f"expected Marks: {prediction[0]:.2f}")

    # visualization
    plt.scatter(df["Hours"], df["Marks"])
    plt.xlabel("study Hours")
    plt.ylabel("Marks")
    plt.title("student performance analysis")
    plt.show()