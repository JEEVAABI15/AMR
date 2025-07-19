import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
import os

st.set_page_config(layout="wide", page_title="AMR Resistance Trend Dashboard")

# Load data with caching for performance
@st.cache_data
def load_data():
    df = pd.read_csv("processed_amr_long.csv")
    return df

df = load_data()

# Sidebar - Filters
st.sidebar.title("🔬 AMR Filter Options")

species_list = ["All"] + sorted(df['Species'].dropna().unique())
country_list = ["All"] + sorted(df['Country'].dropna().unique())
antibiotics_list = ["All"] + sorted(df['Antibiotic'].dropna().unique())
year_range = sorted(df['Year'].dropna().unique())

default_species = ["All"]
default_antibiotic = "All"
default_country = "All"
selected_species = st.sidebar.multiselect("🦠 Select Species", species_list, default=default_species)
selected_antibiotic = st.sidebar.selectbox("💊 Select Antibiotic", antibiotics_list, index=0)
selected_country = st.sidebar.selectbox("🌍 Select Country", country_list, index=0)

if len(year_range) > 1:
    selected_years = st.sidebar.slider(
        "📅 Year Range",
        min_value=int(min(year_range)),
        max_value=int(max(year_range)),
        value=(int(min(year_range)), int(max(year_range)))
    )
else:
    selected_years = (int(year_range[0]), int(year_range[0]))
    st.sidebar.info(f"Only one year in data: {year_range[0]}")

# Optional gene filter
with st.sidebar.expander("🧬 Filter by Resistance Genes"):
    gene_columns = [col for col in df.columns if col in ["NDM", "KPC", "OXA", "VIM", "IMP", "SPM", "GIM"]]
    selected_genes = [gene for gene in gene_columns if st.checkbox(gene, False)]

# Filtering logic with 'All' options
species_filter = df['Species'].isin(selected_species) if "All" not in selected_species else df['Species'].notna()
antibiotic_filter = (df['Antibiotic'] == selected_antibiotic) if selected_antibiotic != "All" else df['Antibiotic'].notna()
country_filter = (df['Country'] == selected_country) if selected_country != "All" else df['Country'].notna()
year_filter = df['Year'].between(selected_years[0], selected_years[1])

filtered_df = df[species_filter & antibiotic_filter & country_filter & year_filter].copy()

if selected_genes:
    for gene in selected_genes:
        filtered_df = filtered_df[filtered_df[gene] == 1]

# Debugging: Show number of filtered rows and a table
st.write(f"Filtered rows: {len(filtered_df)}")
if isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty:
    st.dataframe(filtered_df.head(20))

st.title("🧪 Antimicrobial Resistance Dashboard")
st.markdown(f"""
This dashboard shows resistance trends for **{selected_antibiotic if selected_antibiotic != 'All' else 'all antibiotics'}** across selected species and regions.
""")

# Resistance trend over years
st.subheader("📈 Resistance Rate Over Time")

if isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty:
    trend_data = (
        filtered_df.groupby(['Year'])['Resistance_Binary']
        .mean()
        .reset_index()
        .rename(columns={'Resistance_Binary': 'Resistance Rate'})
    )
    fig, ax = plt.subplots()
    sns.lineplot(data=trend_data, x='Year', y='Resistance Rate', marker='o', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Resistance Rate (0=Susceptible, 1=Resistant)")
    ax.set_title(f"Resistance Trend of {selected_antibiotic if selected_antibiotic != 'All' else 'All Antibiotics'} in {selected_country if selected_country != 'All' else 'All Countries'}")
    st.pyplot(fig)
else:
    st.warning("⚠️ No data available for selected filters.")

# S/I/R Pie Chart
st.subheader("🧬 Phenotypic Distribution (S/I/R)")

if isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty:
    pie_data = filtered_df['Interpretation_clean'].value_counts()
    fig2, ax2 = plt.subplots()
    labels = [str(label) for label in pie_data.index]
    ax2.pie(pie_data.values.tolist(), labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    st.pyplot(fig2)
else:
    st.warning("⚠️ Not enough data to generate pie chart.")

# Resistance rate by species (within selected antibiotic and country)
st.subheader("📊 Resistance Rate by Species")
species_data = (
    df[(antibiotic_filter) & (country_filter) & (year_filter)]
    .groupby('Species')['Resistance_Binary'].mean()
)
species_data = pd.Series(species_data).sort_values(ascending=False).head(10)
fig3, ax3 = plt.subplots()
species_data.plot(kind='barh', color='salmon', ax=ax3)
ax3.set_xlabel("Resistance Rate")
ax3.set_title(f"Top 10 Species - Resistance to {selected_antibiotic if selected_antibiotic != 'All' else 'All Antibiotics'} in {selected_country if selected_country != 'All' else 'All Countries'}")
st.pyplot(fig3)

# Age range filter in sidebar
min_age = int(df['Age_Lower'].min())
max_age = int(df['Age_Upper'].max())
selected_age_range = st.sidebar.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
age_filter = (df['Age_Lower'] >= selected_age_range[0]) & (df['Age_Upper'] <= selected_age_range[1])
filtered_df = df[age_filter].copy()

st.subheader("🧒 Resistance Rate by Age Group")
age_group_stats = filtered_df.groupby('Age Group')['Resistance_Binary'].mean().reset_index()
if not age_group_stats.empty:
    st.bar_chart(age_group_stats.set_index('Age Group'))
else:
    st.info("No data for selected age range.")

st.subheader("📈 Resistance Rate by Age (Lower Bound)")
age_bin_stats = filtered_df.groupby('Age_Lower')['Resistance_Binary'].mean().reset_index()
if not age_bin_stats.empty:
    st.line_chart(age_bin_stats.set_index('Age_Lower'))
else:
    st.info("No data for selected age range.")

st.subheader("👩‍🦰👨‍🦱 Resistance Rate by Age Group and Gender")
gender_age_stats = filtered_df.groupby(['Gender', 'Age Group'])['Resistance_Binary'].mean().unstack('Gender')
if gender_age_stats is not None and not gender_age_stats.empty:
    st.bar_chart(gender_age_stats)
else:
    st.info("No data for selected age/gender combination.")

# Add a slider for years of history to use for forecasting
years_back = st.sidebar.slider("Years of history to use for forecasting", min_value=3, max_value=20, value=10)

def forecast_resistance(df, species, antibiotic, country, years_back=10):
    subset = df[
        (df['Species'] == species) &
        (df['Antibiotic'] == antibiotic) &
        (df['Country'] == country)
    ]
    max_year = subset['Year'].max()
    subset = subset[subset['Year'] >= max_year - years_back + 1]
    yearly = subset.groupby('Year')['Resistance_Binary'].mean().reset_index()
    yearly = yearly.dropna()
    yearly.columns = ['ds', 'y']
    yearly['ds'] = pd.to_datetime(yearly['ds'], format='%Y')
    # Apply 3-year rolling average to smooth
    yearly['y'] = yearly['y'].rolling(window=3, min_periods=1).mean()
    if len(yearly) < 2:
        st.warning('Not enough data to forecast.')
        return
    last_year = yearly['ds'].dt.year.max()
    years_ahead = max(0, 2030 - last_year)
    model = Prophet()
    model.fit(yearly)
    future = model.make_future_dataframe(periods=years_ahead, freq='Y')
    forecast = model.predict(future)
    fig = model.plot(forecast)
    plt.title(f'Forecasted Resistance Rate for {species}, {antibiotic}, {country}')
    plt.xlabel('Year')
    plt.ylabel('Resistance Rate')
    st.pyplot(fig)

def forecast_country_antibiotic(df, antibiotic, country, years_back=10):
    subset = df[
        (df['Antibiotic'] == antibiotic) &
        (df['Country'] == country)
    ]
    max_year = subset['Year'].max()
    subset = subset[subset['Year'] >= max_year - years_back + 1]
    yearly = subset.groupby('Year')['Resistance_Binary'].mean().reset_index()
    yearly = yearly.dropna()
    yearly.columns = ['ds', 'y']
    yearly['ds'] = pd.to_datetime(yearly['ds'], format='%Y')
    # Apply 3-year rolling average to smooth
    yearly['y'] = yearly['y'].rolling(window=3, min_periods=1).mean()
    if len(yearly) < 2:
        st.warning('Not enough data to forecast.')
        return
    last_year = yearly['ds'].dt.year.max()
    years_ahead = max(0, 2030 - last_year)
    model = Prophet()
    model.fit(yearly)
    future = model.make_future_dataframe(periods=years_ahead, freq='Y')
    forecast = model.predict(future)
    fig = model.plot(forecast)
    plt.title(f'Forecasted Resistance Rate for {antibiotic} in {country} (All Species)')
    plt.xlabel('Year')
    plt.ylabel('Resistance Rate')
    st.pyplot(fig)

# Add forecasting section after main plots
st.subheader("🔮 Forecast Resistance Rate (Time Series)")
if (
    selected_antibiotic != "All" and
    selected_country != "All" and
    isinstance(selected_species, list) and len(selected_species) == 1 and selected_species[0] != "All"
):
    if st.button('Forecast Resistance Rate for Selection'):
        forecast_resistance(df, selected_species[0], selected_antibiotic, selected_country, years_back=years_back)
elif (
    selected_antibiotic != "All" and
    selected_country != "All" and
    isinstance(selected_species, list) and len(selected_species) == 1 and selected_species[0] == "All"
):
    if st.button('Forecast for Country + Antibiotic (All Species)'):
        forecast_country_antibiotic(df, selected_antibiotic, selected_country, years_back=years_back)
else:
    st.info("Please select a valid combination to enable forecasting.")

# Find all valid forecasting combinations (at least 2 years of data)
combo_counts = (
    df.groupby(['Species', 'Antibiotic', 'Country'])['Year']
    .nunique()
    .reset_index()
    .rename(columns={'Year': 'YearCount'})
)
valid_combos = combo_counts[combo_counts['YearCount'] >= 2].copy()

# Only show the combination(s) with the maximum number of years
if not valid_combos.empty:
    max_years = valid_combos['YearCount'].max()
    max_combos = valid_combos[valid_combos['YearCount'] == max_years]
    max_combos = pd.DataFrame(max_combos)  # Ensure DataFrame for downstream operations
else:
    max_combos = valid_combos

st.subheader("💡 Forecast Suggestions (Most Robust)")
if not max_combos.empty:
    st.write(f"These combination(s) have the maximum number of years ({max_years}) and are most robust for forecasting:")
    suggestion = st.selectbox(
        "Select a combination to forecast:",
        max_combos.apply(lambda row: f"{row['Species']} | {row['Antibiotic']} | {row['Country']} ({row['YearCount']} years)", axis=1)
    )
    if st.button('Forecast Selected Suggestion'):
        selected = max_combos.iloc[list(max_combos.apply(lambda row: f"{row['Species']} | {row['Antibiotic']} | {row['Country']} ({row['YearCount']} years)", axis=1)).index(suggestion)]
        forecast_resistance(df, selected['Species'], selected['Antibiotic'], selected['Country'], years_back=years_back)
else:
    st.info("No valid combinations with at least 2 years of data found.")

# Footer
st.markdown("---")
st.markdown("📊 Built by Jeeva Abishake for Vivli AMR Challenge")

# st.write("Current working directory:", os.getcwd())
# st.write("Columns in df:", df.columns.tolist()) 