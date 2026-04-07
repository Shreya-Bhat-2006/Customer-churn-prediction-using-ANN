import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn

# ===================== DATA LOADING =====================
df = pd.read_csv("Data.csv")

# ===================== DATA CLEANING =====================
df.drop('customerID', axis="columns", inplace=True)

df = df[df.TotalCharges != " "]
df.TotalCharges = pd.to_numeric(df.TotalCharges)

df.replace('No internet service', 'No', inplace=True)
df.replace('No phone service', 'No', inplace=True)

# ===================== ENCODING =====================
yes_no_columns = [
    'Partner','Dependents','PhoneService','MultipleLines',
    'OnlineSecurity','OnlineBackup','DeviceProtection',
    'TechSupport','StreamingTV','StreamingMovies',
    'PaperlessBilling','Churn'
]

for col in yes_no_columns:
    df[col] = df[col].map({"Yes": 1, "No": 0})

df["gender"] = df["gender"].map({"Male": 0, "Female": 1})

df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], dtype=int)

# ===================== SCALING =====================
col_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]

scaler = MinMaxScaler()
df[col_to_scale] = scaler.fit_transform(df[col_to_scale])

print(df["Churn"].value_counts())

# ===================== SPLIT =====================
X = df.drop("Churn", axis="columns")
Y = df["Churn"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=5
)

# ===================== MODEL =====================
model = keras.Sequential([
    keras.layers.Dense(32, input_shape=(X_train.shape[1],), activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ===================== TRAINING =====================
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    Y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stop],
    class_weight={0:1, 1:2}
)

# ===================== LOSS GRAPH =====================
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend(['Train Loss', 'Validation Loss'])

plt.show()
plt.close()

# ===================== EVALUATION =====================
model.evaluate(X_test, Y_test)

# ===================== FINAL PREDICTION =====================
y_probs = model.predict(X_test)

# FINAL THRESHOLD (BEST FOUND)
final_threshold = 0.4

Y_pred = (y_probs > final_threshold).astype(int).flatten()

print("\nFINAL MODEL (Threshold = 0.4)")
print(classification_report(Y_test, Y_pred))

# ===================== CONFUSION MATRIX =====================
cm = confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix (Final Model)")
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()
plt.close()

# ===================== (OPTIONAL) SAVE MODEL =====================
# model.save("churn_model.h5")