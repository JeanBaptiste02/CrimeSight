# Crime Data Analysis Dashboard

This project provides a comprehensive analysis of crime data with an interactive web dashboard for visualization.

## Features

- Data preprocessing and cleaning
- Descriptive statistics
- Interactive visualizations
- Advanced insights and analysis
- Web-based dashboard

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Place your crime data CSV file in the root directory as `crime_data.csv`

3. Run the analysis script:
```bash
python crime_analysis.py
```

4. Open `index.html` in a web browser to view the dashboard

## Output Files

The analysis generates several output files:

- `cleaned_crime_data.csv`: Preprocessed and cleaned dataset
- `crime_statistics.json`: Descriptive statistics
- `crime_relationships.json`: Relationship analysis
- `crime_insights.json`: Advanced insights
- `visualizations/`: Directory containing all generated visualizations

## Dashboard Sections

1. Overview Statistics
   - Total record count
   - Categorical value distributions

2. Visualizations
   - Age distributions
   - Gender distributions
   - Race distributions
   - Crime category distributions
   - Fatal status distributions

3. Advanced Insights
   - Top crime categories by victim age
   - Violence crime demographics
   - Victim-offender relationships

## Data Requirements

The input CSV file should contain the following columns:
- Disposition
- OffenderStatus
- Offender_Race
- Offender_Gender
- Offender_Age
- PersonType
- Victim_Race
- Victim_Gender
- Victim_Age
- Victim_Fatal_Status
- Report Type
- Category

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- scikit-learn
- HTML/CSS/JavaScript
- Bootstrap 