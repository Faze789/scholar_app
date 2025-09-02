import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
df = pd.read_csv('nust_merit_data.csv')

# Encode categorical columns
le_program = LabelEncoder()
le_institute = LabelEncoder()

df['Program_enc'] = le_program.fit_transform(df['Program'])
df['Institute_enc'] = le_institute.fit_transform(df['Institute'])

# Features: Program, Institute
X = df[['Program_enc', 'Institute_enc']].values
y = df['Closing (%)'].values

# Build and train the model
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=0)

# List available programs
programs = df['Program'].unique()
print("Available Programs:")
for idx, program in enumerate(programs):
    print(f"{idx+1}. {program}")

# Ask user for program choice
prog_choice = int(input("Enter the number of the program you want to apply for: ")) - 1
selected_program = programs[prog_choice]

# List available institutes for the selected program
institutes = df[df['Program'] == selected_program]['Institute'].unique()
print("Available Institutes for this program:")
for idx, inst in enumerate(institutes):
    print(f"{idx+1}. {inst}")

inst_choice = int(input("Enter the number of the institute: ")) - 1
selected_institute = institutes[inst_choice]

# Encode user choices
prog_enc = le_program.transform([selected_program])[0]
inst_enc = le_institute.transform([selected_institute])[0]

# Predict closing merit for 2026
x_pred = np.array([[prog_enc, inst_enc]])
pred_merit = model.predict(x_pred, verbose=0)[0][0]

# Ask user for their marks
matric_marks = float(input("Enter your Matric marks (out of 1100): "))
fsc_marks = float(input("Enter your FSc marks (out of 1100): "))
net_marks = float(input("Enter your NET marks (out of 200): "))

# Calculate aggregate
matric_perc = (matric_marks / 1100) * 100
fsc_perc = (fsc_marks / 1100) * 100
net_perc = (net_marks / 200) * 100

aggregate = (matric_perc * 0.10) + (fsc_perc * 0.15) + (net_perc * 0.75)

print(f"\nPredicted closing merit for 2026 ({selected_program}, {selected_institute}): {pred_merit:.2f}")
print(f"Your calculated aggregate: {aggregate:.2f}")

if aggregate >= pred_merit:
    print("You have a chance to get admission!")
else:
    print("Chances are low, but you can still apply.")