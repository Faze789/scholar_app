import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the CSV file
df = pd.read_csv('comsats_merit_data.csv')

# Encode categorical columns
le_program = LabelEncoder()
le_campus = LabelEncoder()
le_year = LabelEncoder()

df['Program_enc'] = le_program.fit_transform(df['Program'])
df['Campus_enc'] = le_campus.fit_transform(df['Campus'])
df['Year_enc'] = le_year.fit_transform(df['Year'])

# Features and target
X = df[['Program_enc', 'Campus_enc', 'Year_enc']].values
y = df['Closing Merit (%)'].values

# Build a simple neural network model
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=200, verbose=0)

# Predict merit for 2026 for each program
year_2026_enc = le_year.transform([max(df['Year'].unique())])[0] + 1  # Next year encoding

# List available programs
programs = df['Program'].unique()
print("Available Programs:")
for idx, program in enumerate(programs):
    print(f"{idx+1}. {program}")

# Ask user for program choice
prog_choice = int(input("Enter the number of the program you want to apply for: ")) - 1
selected_program = programs[prog_choice]

# Predict merit for selected program
prog_enc = le_program.transform([selected_program])[0]
campus = 'Islamabad'
campus_enc = le_campus.transform([campus])[0]
x_pred = np.array([[prog_enc, campus_enc, year_2026_enc]])
pred_merit = model.predict(x_pred, verbose=0)[0][0]

# Ask user for their aggregate
user_agg = float(input("Enter your aggregate (%): "))

print(f"\nPredicted closing merit for 2026 ({selected_program}, Islamabad campus): {pred_merit:.2f}")
if user_agg >= pred_merit:
    print("You have a chance!")
else:
    print("You may not have a chance, but you can still apply.")