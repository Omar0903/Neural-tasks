import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tkinter import messagebox
import matplotlib.pyplot as plt



def signum(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


def PerceptronTrain(X, T, eta=0.01, epochs=10, useBias=True):
    numberOfSamples = len(X)
    numberOfFeatures = len(X[0])
    Weights = np.random.randn(numberOfFeatures)
    bias = np.random.randn() if useBias else 0

    for _ in range(epochs):
        for i in range(numberOfSamples):
            Output = signum(np.dot(Weights, X[i]) + bias)
            if Output != T[i]:
                Weights += eta * (T[i] - Output) * X[i]
                if useBias:
                    bias += eta * (T[i] - Output)
    return Weights, bias


def AdalineTrain(X, T, eta=0.01, epochs=100, mseThreshold=0.001, useBias=True):
    numberOfSamples = len(X)
    numberOfFeatures = len(X[0])
    Weights = np.random.randn(numberOfFeatures)
    bias = np.random.randn() if useBias else 0

    for epoch in range(epochs):
        Y = np.dot(X, Weights) + bias
        errors = T - Y
        mse = np.mean(errors ** 2)
        if mse <= mseThreshold:
            break
        Weights += eta * np.dot(X.T, errors) / numberOfSamples
        if useBias:
            bias += eta * np.mean(errors)
    return Weights, bias, mse

dataFrame = pd.read_csv("penguins.csv")


dataFrame['CulmenLength']=dataFrame['CulmenLength'].fillna(dataFrame['CulmenLength'].mean())
dataFrame['CulmenDepth']=dataFrame['CulmenDepth'].fillna(dataFrame['CulmenDepth'].mean())
dataFrame['BodyMass']=dataFrame['BodyMass'].fillna(dataFrame['BodyMass'].mean())
dataFrame['FlipperLength']=dataFrame['FlipperLength'].fillna(dataFrame['FlipperLength'].mean())


labelEncoding = LabelEncoder()
dataFrame["OriginLocation"] = labelEncoding.fit_transform(dataFrame["OriginLocation"])
dataFrame["Target"] = LabelEncoder().fit_transform(dataFrame["Species"])


featureMap = {
    'Culmen Length and Culmen Depth': ('CulmenLength', 'CulmenDepth'),
    'Culmen Length and Flipper Length': ('CulmenLength', 'FlipperLength'),
    'Culmen Length and Body Mass': ('CulmenLength', 'BodyMass'),
    'Culmen Length and Origin Location': ('CulmenLength', 'OriginLocation'),
    'Culmen Depth and Flipper Length': ('CulmenDepth', 'FlipperLength'),
    'Culmen Depth and Body Mass': ('CulmenDepth', 'BodyMass'),
    'Culmen Depth and Origin Location': ('CulmenDepth', 'OriginLocation'),
    'Flipper Length and Body Mass': ('FlipperLength', 'BodyMass'),
    'Flipper Length and Origin Location': ('FlipperLength', 'OriginLocation'),
    'Origin Location and Body Mass': ('OriginLocation', 'BodyMass'),
}

classMap = {
    'Adelie and Gentoo': (0, 1),
    'Adelie and Chinstrap': (0, 2),
    'Chinstrap and Gentoo': (1, 2),
}





def plotDecisionBoundary(X, T, W, b, title="Decision Boundary"):
    if X.shape[1] != 2:
        raise ValueError("X must have exactly two features for 2D plotting")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = -(W[0] * x_vals + b) / W[1]

    plt.figure(figsize=(8, 6))
    plt.scatter(X[T == -1, 0], X[T == -1, 1], color='red', label='Class -1')
    plt.scatter(X[T == 1, 0], X[T == 1, 1], color='blue', label='Class 1')
    plt.plot(x_vals, y_vals, 'k--', linewidth=2, label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def RunModel(En1, En2, En3, cmbo1, cmbo2, cmbo3, biasOption):
    try:
        epochs = int(En1.get())
        mseThreshold = float(En2.get())
        eta = float(En3.get())
        selectFeature = cmbo1.get()
        selectClasses = cmbo2.get()
        algorithm = cmbo3.get()
        useBias = (biasOption.get() == "Yes")

        featureOne, featureTwo = featureMap[selectFeature]
        ClassOne, ClassTwo = classMap[selectClasses]

        subset = dataFrame[(dataFrame["Target"] == ClassOne) | (dataFrame["Target"] == ClassTwo)]

        train_ClassOne = subset[subset["Target"] == ClassOne].sample(n=30, random_state=42)
        test_ClassOne = subset[subset["Target"] == ClassOne].drop(train_ClassOne.index)
        train_ClassTwo = subset[subset["Target"] == ClassTwo].sample(n=30, random_state=42)
        test_ClassTwo = subset[subset["Target"] == ClassTwo].drop(train_ClassTwo.index)

        train_df = pd.concat([train_ClassOne, train_ClassTwo]).sample(frac=1, random_state=42)
        test_df = pd.concat([test_ClassOne, test_ClassTwo]).sample(frac=1, random_state=42)

        train_df["Target"] = train_df["Target"].apply(lambda x: 1 if x == ClassOne else -1)
        test_df["Target"] = test_df["Target"].apply(lambda x: 1 if x == ClassOne else -1)

        X_train = train_df[[featureOne, featureTwo]].values.astype(float)
        T_train = train_df["Target"].values
        X_test = test_df[[featureOne, featureTwo]].values.astype(float)
        T_test = test_df["Target"].values

        X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
        X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

        if algorithm == "Perceptron":
            W, b = PerceptronTrain(X_train, T_train, eta=eta, epochs=epochs, useBias=useBias)
            mse = None
        else:
            W, b, mse = AdalineTrain(X_train, T_train, eta=eta, epochs=epochs,
                                      mseThreshold=mseThreshold, useBias=useBias)

        preds = np.array([signum(np.dot(W, x_i) + b) for x_i in X_test])
        acc = np.mean(preds == T_test)

        msg = f"Dataset: class_{ClassOne}_{ClassTwo}\nFeatures: ({featureOne}, {featureTwo})\nAccuracy: {acc*100:.2f}%"
        if algorithm == "Adaline" and mse is not None:
            msg += f"\nFinal MSE: {mse:.6f}"
        messagebox.showinfo("Result", msg)

        plotDecisionBoundary(X_train, T_train, W, b, title=f"{algorithm} Decision Boundary")

    except Exception as e:
        messagebox.showerror("Error", str(e))
