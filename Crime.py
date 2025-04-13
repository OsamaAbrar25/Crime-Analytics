import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, mean_absolute_error, confusion_matrix

# Set plot style for better visuals
sns.set_style("whitegrid")

# Load the dataset
# Assuming the dataset is saved as 'murder_victims.csv'
# If you need to read from a string or upload, let me know!
df = pd.read_csv("C:\\Users\\osama\\OneDrive\\Desktop\\Python Project\\Murder_victim_age_sex.csv")  # Replace with correct path if needed

# Display basic info and first few rows
print("Dataset Info:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())

# Data Cleaning

# Replace 'NULL' with NaN for proper handling
df.replace('NULL', pd.NA, inplace=True)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values
# For numeric columns, impute with 0 (assuming no victims if missing)
numeric_cols = ['Victims_Above_50_Yrs', 'Victims_Total', 'Victims_Upto_10_15_Yrs',
                'Victims_Upto_10_Yrs', 'Victims_Upto_15_18_Yrs', 'Victims_Upto_18_30_Yrs',
                'Victims_Upto_30_50_Yrs']
for col in numeric_cols:
    df[col] = df[col].fillna(0).astype(int)  # Convert to integer after filling

# Verify no missing values remain in numeric columns
print("\nMissing Values After Imputation:")
print(df[numeric_cols].isnull().sum())

# Check for duplicates
print("\nDuplicate Rows:", df.duplicated().sum())
# If duplicates exist, drop them
df.drop_duplicates(inplace=True)

# Ensure correct data types
df['Year'] = df['Year'].astype(int)
df['Area_Name'] = df['Area_Name'].astype(str)
df['Group_Name'] = df['Group_Name'].astype(str)

# Drop redundant columns (Group_Name and Sub_Group_Name are consistent within gender)
df = df.drop(['Sub_Group_Name'], axis=1)

# Exploratory Data Analysis (EDA)

# Summary statistics
print("\nSummary Statistics:")
print(df[numeric_cols].describe())

# 3.1: Total Victims by Year
total_by_year = df.groupby('Year')['Victims_Total'].sum()
plt.figure(figsize=(10, 6))
total_by_year.plot(kind='line', marker='o')
plt.title('Total Murder Victims by Year (2001-2010)')
plt.xlabel('Year')
plt.ylabel('Total Victims')
plt.savefig('total_victims_by_year.png')  # Save plot for your project
plt.show()

# 3.2: Total Victims by State (Top 10)
total_by_state = df[df['Group_Name'] == 'Murder - Total Victims'].groupby('Area_Name')['Victims_Total'].sum()
top_10_states = total_by_state.sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_states.values, y=top_10_states.index)
plt.title('Top 10 States by Total Murder Victims (2001-2010)')
plt.xlabel('Total Victims')
plt.ylabel('State/UT')
plt.savefig('top_10_states.png')
plt.show()

# 3.3: Gender Comparison
gender_data = df[df['Group_Name'].isin(['Murder - Male Victims', 'Murder - Female Victims'])]
gender_by_year = gender_data.groupby(['Year', 'Group_Name'])['Victims_Total'].sum().unstack()
plt.figure(figsize=(10, 6))
gender_by_year.plot(kind='line', marker='o')
plt.title('Male vs Female Murder Victims by Year')
plt.xlabel('Year')
plt.ylabel('Total Victims')
plt.legend(['Female Victims', 'Male Victims'])
plt.savefig('gender_comparison.png')
plt.show()

# 3.4: Age Group Distribution (for Total Victims in 2010)
age_cols = ['Victims_Upto_10_Yrs', 'Victims_Upto_10_15_Yrs', 'Victims_Upto_15_18_Yrs',
            'Victims_Upto_18_30_Yrs', 'Victims_Upto_30_50_Yrs', 'Victims_Above_50_Yrs']
age_data_2010 = df[(df['Year'] == 2010) & (df['Group_Name'] == 'Murder - Total Victims')][age_cols].sum()
plt.figure(figsize=(10, 6))
sns.barplot(x=age_data_2010.index, y=age_data_2010.values)
plt.title('Age Distribution of Murder Victims (2010)')
plt.xlabel('Age Group')
plt.ylabel('Number of Victims')
plt.xticks(rotation=45)
plt.savefig('age_distribution_2010.png')
plt.show()

# 3.5: Correlation Heatmap for Age Groups
corr_matrix = df[age_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Between Age Groups of Victims')
plt.savefig('age_correlation.png')
plt.show()

print("\nEDA Complete! Plots saved as PNG files.")




# Filter for "Murder - Total Victims"
df_total = df[df['Group_Name'] == 'Murder - Total Victims'].copy()

# Features (exclude Male_Victims and Female_Victims to avoid leakage)
features = ['Year', 'Victims_Upto_10_Yrs', 'Victims_Upto_10_15_Yrs', 'Victims_Upto_15_18_Yrs',
            'Victims_Upto_18_30_Yrs', 'Victims_Upto_30_50_Yrs', 'Victims_Above_50_Yrs']
X = df_total[features]

# Classification Task (Random Forest)
# Create binary target: High (1) if Victims_Total > median, else Low (0)
median_total = df_total['Victims_Total'].median()
df_total['High_Victims'] = (df_total['Victims_Total'] > median_total).astype(int)
y_class = df_total['High_Victims']

# Split data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

# Train Random Forest Classifier with constraints to reduce overfitting
clf = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42)
clf.fit(X_train_c_scaled, y_train_c)

# Predict and evaluate
y_pred_c = clf.predict(X_test_c_scaled)
print("\nClassification Results (Random Forest):")
print("Accuracy:", accuracy_score(y_test_c, y_pred_c))
print("Precision:", precision_score(y_test_c, y_pred_c))
print("Recall:", recall_score(y_test_c, y_pred_c))
print("F1-Score:", f1_score(y_test_c, y_pred_c))

# Cross-validation to get robust accuracy estimate
cv_scores = cross_val_score(clf, scaler.fit_transform(X), y_class, cv=5, scoring='accuracy')
print("\nCross-Validation Accuracy: Mean =", cv_scores.mean(), "Std =", cv_scores.std())

# Compute confusion matrix
cm = confusion_matrix(y_test_c, y_pred_c)

# Plot confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('rf_confusion.png')
plt.show()

# Plot feature importance
feature_importance = pd.Series(clf.feature_importances_, index=features)
plt.figure(figsize=(10, 6))
feature_importance.sort_values().plot(kind='barh')
plt.title('Feature Importance for Random Forest Classifier')
plt.xlabel('Importance')
plt.savefig('rf_feature_importance.png')
plt.show()

# Regression Task (SVR)
# Prepared data with original features (including Male_Victims, Female_Victims for regression)
df_male = df[df['Group_Name'] == 'Murder - Male Victims'][['Area_Name', 'Year', 'Victims_Total']].rename(columns={'Victims_Total': 'Male_Victims'})
df_female = df[df['Group_Name'] == 'Murder - Female Victims'][['Area_Name', 'Year', 'Victims_Total']].rename(columns={'Victims_Total': 'Female_Victims'})
df_model = df_total.merge(df_male, on=['Area_Name', 'Year'], how='left').merge(df_female, on=['Area_Name', 'Year'], how='left')
df_model['Male_Victims'] = df_model['Male_Victims'].fillna(0).astype(int)
df_model['Female_Victims'] = df_model['Female_Victims'].fillna(0).astype(int)

# Features for regression
features_reg = ['Year', 'Victims_Upto_10_Yrs', 'Victims_Upto_10_15_Yrs', 'Victims_Upto_15_18_Yrs',
                'Victims_Upto_18_30_Yrs', 'Victims_Upto_30_50_Yrs', 'Victims_Above_50_Yrs',
                'Male_Victims', 'Female_Victims']
X_reg = df_model[features_reg]
y_reg = df_model['Victims_Total']

# Split data
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Standardize features
X_train_r_scaled = scaler.fit_transform(X_train_r)
X_test_r_scaled = scaler.transform(X_test_r)

# Train SVR
reg = SVR(kernel='rbf', C=100, epsilon=0.1)
reg.fit(X_train_r_scaled, y_train_r)

# Predict and evaluate
y_pred_r = reg.predict(X_test_r_scaled)
print("\nRegression Results (SVR):")
print("Mean Squared Error:", mean_squared_error(y_test_r, y_pred_r))
print("Mean Absolute Error:", mean_absolute_error(y_test_r, y_pred_r))
print("R-squared:", r2_score(y_test_r, y_pred_r))

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test_r, y_pred_r, alpha=0.5)
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
plt.title('Actual vs Predicted Victims_Total (SVR)')
plt.xlabel('Actual Victims')
plt.ylabel('Predicted Victims')
plt.savefig('svr_predictions.png')
plt.show()










































