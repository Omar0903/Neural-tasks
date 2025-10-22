import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import itertools
dataFrame = pd.read_csv("penguins.csv")

df = pd.read_csv("penguins.csv")
df['CulmenLength']=df['CulmenLength'].fillna(df['CulmenLength'].mean())
df['CulmenDepth']=df['CulmenDepth'].fillna(df['CulmenDepth'].mean())
df['BodyMass']=df['BodyMass'].fillna(df['BodyMass'].mean())
df['FlipperLength']=df['FlipperLength'].fillna(df['FlipperLength'].mean())
le = LabelEncoder()
df["OriginLocation"] = le.fit_transform(df["OriginLocation"])
df = df.iloc[:150].reset_index(drop=True)
df["Target"] = [0]*50 + [1]*50 + [2]*50
feature_names = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass", "OriginLocation"]
def signum(x):
    return 1 if x >= 0 else -1
def perceptron_train(X, T, eta=0.01, epochs=20):
    n_samples, n_features = X.shape
    W = np.random.randn(n_features)
    b = np.random.randn()
    for _ in range(epochs):
        for i in range(n_samples):
            y_i = signum(np.dot(W, X[i]) + b)
            if y_i != T[i]:
                W += eta * (T[i] - y_i) * X[i]
                b += eta * (T[i] - y_i)
    return W, b
feature_pairs = list(itertools.combinations(feature_names, 2))
class_pairs = list(itertools.combinations([0,1,2], 2))

for (c1, c2) in class_pairs:
    subset = df[(df["Target"] == c1) | (df["Target"] == c2)]
    print(f"\nDataset: class_{c1}_{c2}")
    train_c1 = subset[subset["Target"] == c1].sample(n=30, random_state=42)
    test_c1  = subset[subset["Target"] == c1].drop(train_c1.index)
    train_c2 = subset[subset["Target"] == c2].sample(n=30, random_state=42)
    test_c2  = subset[subset["Target"] == c2].drop(train_c2.index)
    train_df = pd.concat([train_c1, train_c2]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df  = pd.concat([test_c1, test_c2]).sample(frac=1, random_state=42).reset_index(drop=True)
    train_df["Target"] = train_df["Target"].apply(lambda x: 1 if x == c1 else -1)
    test_df["Target"]  = test_df["Target"].apply(lambda x: 1 if x == c1 else -1)
    for f1, f2 in feature_pairs:
        X_train = train_df[[f1, f2]].values.astype(float)
        T_train = train_df["Target"].values
        X_test  = test_df[[f1, f2]].values.astype(float)
        T_test  = test_df["Target"].values
        W, b = perceptron_train(X_train, T_train, eta=0.01, epochs=10)
        preds = np.array([signum(np.dot(W, x_i) + b) for x_i in X_test])
        acc = np.mean(preds == T_test)
        print(f"   → Features: ({f1}, {f2}) | Test Accuracy: {acc*100:.2f}%")
