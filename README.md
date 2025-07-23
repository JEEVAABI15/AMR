# AMR Resistance Forecasting Dashboard

This project provides an interactive dashboard for exploring, analyzing, and forecasting antimicrobial resistance (AMR) trends using the ATLAS dataset.

## 🚀 Features
- Visualize resistance rates by species, antibiotic, country, year, age group, and gender
- Filter by resistance genes, age range, and more
- Forecast resistance rates up to 2030 using Prophet time series modeling
- Demographic insights: resistance by age and gender
- Robust preprocessing pipeline for large AMR datasets

## 📦 Setup Instructions
1. **Clone the repository and navigate to the project directory:**
   ```bash
   git clone <your-repo-url>
   cd AMR
   ```
2. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   # or
   python3 -m pip install -r requirements.txt
   ```
3. **Preprocess the data:**
   ```bash
   python3 data_preprocessing.py
   ```
   This will generate `processed_amr_long.csv` for use in the dashboard.

4. **Run the dashboard:**
   ```bash
   streamlit run dashboard_app.py
   ```
   Open the provided local URL in your browser.

## 📝 Usage
- Use the sidebar to filter by species, antibiotic, country, year, age, gender, and resistance genes.
- Explore resistance trends, demographic breakdowns, and S/I/R distributions.
- Use the forecasting section to predict resistance rates up to 2030 for any valid combination.
- See "Forecast Suggestions" for the most robust time series combinations.

## 📊 Example Visualizations
- Resistance rate by age group (bar chart)
- Resistance rate by year (line chart)
- S/I/R distribution (pie chart)
- Resistance by age group and gender (grouped bar chart)
- Forecasted resistance rate with confidence intervals (Prophet)

## 🧬 Data Source
- [Pfizer ATLAS Dataset](https://atlas-surveillance.com/)

## 🤝 Contributors
- Jeeva Abishake
- [Your Name Here]

## 📄 License
This project is open source and free to use for research and educational purposes. 