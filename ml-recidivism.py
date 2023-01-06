import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./recidivism.csv')
print(data.head())
print(data.shape)


# Plotting race and Convicting Offense Classification to understand the data more 
plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1) # row 1, col 2 index 1
data['Race - Ethnicity'].value_counts().plot(kind='bar')
plt.title("Race - Ethnicity")
plt.ylabel('Count')

plt.subplot(1, 2, 2) # index 2
data['Convicting Offense Classification'].value_counts().plot(kind='bar')
plt.title("Convicting Offense Classification")
plt.show()

# Plotting age and Recidivism  
plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1) # row 1, col 2 index 1
data['Recidivism - Return to Prison numeric'].value_counts().plot(kind='bar')
plt.title("Recidivism - Return to Prison numeric")
plt.ylabel('Count')

plt.subplot(1, 2, 2) # index 2
data['Age At Release '].value_counts().plot(kind='bar')
plt.title("Age At Release ")
plt.show()


# Seeing if there is a correlation between age and recidivism 
plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1) # row 1, col 2 index 1
data.loc[data['Recidivism - Return to Prison numeric'] == 1]['Age At Release '].value_counts().plot(kind='bar')
plt.title("Age at release of people who returned to prison")
plt.ylabel('Count')

plt.subplot(1, 2, 2) # index 2
data.loc[data['Recidivism - Return to Prison numeric'] == 0]['Age At Release '].value_counts().plot(kind='bar')
plt.title("Age at release of people who didn't return")
plt.show()

# Cleaning up the data. Here we create new columns for categorical columns like Race, Age, Conviction Offense, and Release type. 
# Machine learning algorithms prefer numbers as input. 
data_race = pd.get_dummies(data['Race - Ethnicity'])
placeholder_data = pd.concat([data, data_race], axis=1)

data_age = pd.get_dummies(placeholder_data['Age At Release '])
placeholder_data = pd.concat([placeholder_data, data_age], axis=1)

data_offese_clasification = pd.get_dummies(placeholder_data['Convicting Offense Classification'])
placeholder_data = pd.concat([placeholder_data, data_offese_clasification], axis=1)

data_release_type = pd.get_dummies(placeholder_data['Release Type'])
placeholder_data = pd.concat([placeholder_data, data_release_type], axis=1)

final_data = placeholder_data.drop(["Fiscal Year Released", "Recidivism Reporting Year", "Race - Ethnicity", "Age At Release ", "Convicting Offense Classification", "Convicting Offense Type", "Convicting Offense Subtype", "Release Type", "Release type: Paroled to Detainder united", "Main Supervising District", "N/A -", "Part of Target Population"], axis = 1)


# Spliting the data into input data (X), and output data (y). The input variables will be the variables that will be fed into the model and the output
# variable will be the "Recidivism - Return to Prison numeric". 
X = data.drop(["Recidivism - Return to Prison numeric"], axis = 1)
y = data["Recidivism - Return to Prison numeric"]

print(X.head())
print(y.head())