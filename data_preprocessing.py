import pandas as pd
import numpy as np
import os

# Load the dataset
input_path = "atlas_antibiotics.xlsx"
df = pd.read_excel(input_path)

# Preview structure
print("Initial shape:", df.shape)
print("Columns:\n", df.columns.tolist())

# Drop rows with missing essential data
df = df.dropna(subset=['Species', 'Country', 'Year'])

# Identify antibiotic interpretation columns (those ending with _I) and their value columns
value_vars = []
interpretation_vars = []
antibiotic_names = []
for col in df.columns:
    if col.endswith('_I') and col[:-2] in df.columns:
        interpretation_vars.append(col)
        value_vars.append(col[:-2])
        antibiotic_names.append(col[:-2])

print("\nAntibiotic interpretation columns detected:", interpretation_vars[:5], "... (total:", len(interpretation_vars), ")")

# id_vars for melting
id_vars = ['Isolate Id', 'Study', 'Species', 'Family', 'Country', 'State', 'Gender',
           'Age Group', 'Speciality', 'Source', 'In / Out Patient', 'Year', 'Phenotype']

# Melt value columns
df_value = df.melt(
    id_vars=id_vars,
    value_vars=value_vars,
    var_name='Antibiotic',
    value_name='Value'
)

# Melt interpretation columns
df_interp = df.melt(
    id_vars=id_vars,
    value_vars=interpretation_vars,
    var_name='Antibiotic_I',
    value_name='Interpretation'
)
df_interp['Antibiotic'] = df_interp['Antibiotic_I'].str.replace('_I', '', regex=False)

# Merge on all id_vars + Antibiotic
df_long = pd.merge(
    df_value,
    df_interp[id_vars + ['Antibiotic', 'Interpretation']],
    on=id_vars + ['Antibiotic'],
    how='inner'
)

# Clean interpretation values
interpret_map = {
    'S': 'S', 'SUSCEPTIBLE': 'S',
    'I': 'I', 'INTERMEDIATE': 'I',
    'R': 'R', 'RESISTANT': 'R'
}
df_long['Interpretation_clean'] = df_long['Interpretation'].astype(str).str.strip().str.upper().map(interpret_map)

# Drop rows where interpretation is not S/I/R
df_long = df_long[df_long['Interpretation_clean'].isin(['S', 'I', 'R'])].copy()

# Map to binary: S/I = 0, R = 1
df_long['Resistance_Binary'] = (df_long['Interpretation_clean'] == 'R').astype(int)

# Keep Year as numeric (do not convert to datetime)
df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')

# Ensure df_long is a DataFrame
if not isinstance(df_long, pd.DataFrame):
    df_long = pd.DataFrame(df_long)
# Standardize country and species names (keep as Series)
if not isinstance(df_long['Country'], pd.Series):
    df_long['Country'] = pd.Series(df_long['Country'], index=df_long.index)
df_long['Country'] = df_long['Country'].astype(str).str.title()
if not isinstance(df_long['Species'], pd.Series):
    df_long['Species'] = pd.Series(df_long['Species'], index=df_long.index)
df_long['Species'] = df_long['Species'].astype(str).str.title()
if not isinstance(df_long['Age Group'], pd.Series):
    df_long['Age Group'] = pd.Series(df_long['Age Group'], index=df_long.index)

# Parse age group into numeric lower and upper bounds
age_group_col = pd.Series(df_long['Age Group'], index=df_long.index).astype(str).str.strip()
df_long['Age_Lower'] = age_group_col.str.extract(r'(\d+)')[0].astype(float)
age_upper_series = age_group_col.str.extract(r'-(\d+)')[0].astype(float)
df_long['Age_Upper'] = age_upper_series.combine_first(df_long['Age_Lower'])

# Save to CSV for reuse in dashboard/forecasting
output_path = "processed_amr_long.csv"
df_long_out = pd.DataFrame(df_long[['Species','Country','State','Year','Gender','Age Group','Age_Lower','Age_Upper','Antibiotic','Value','Interpretation_clean','Resistance_Binary']])
df_long_out.to_csv(output_path, index=False)
print(f"\n✅ Preprocessed data saved to: {output_path}")
print("Final shape:", df_long.shape) 