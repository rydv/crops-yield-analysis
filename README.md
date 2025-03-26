# Crop Yield Analysis

## Overview
This project analyzes crop yield data in India to explore relationships between various factors and crop yields, identify farm clusters, and provide recommendations for farmers to improve their yields. The analysis is based on a dataset containing 19,689 records of crop production spanning from 1997 to 2020.

## Objectives
- **Data Exploration**: Understand the dataset structure, key statistics, and distributions of crop yields.
- **Correlation Analysis**: Identify relationships between different variables and crop yield.
- **Clustering Analysis**: Group farms based on key features such as area, rainfall, fertilizer, and pesticide usage.
- **Relationship Analysis**: Explore the impact of agricultural inputs on crop yield.
- **Recommendations**: Provide actionable insights for farmers based on the analysis.

## Key Findings
- Production shows the strongest correlation with yield.
- Coconut, Sugarcane, and Banana are the highest-yielding crops in India.
- Five distinct farm clusters were identified based on area, rainfall, fertilizer, and pesticide usage.
- Different crops perform optimally under specific conditions of rainfall, season, and agricultural inputs.

## Reports
- **HTML Report**: A detailed web-based report that includes visualizations and insights from the analysis.
- **PDF Report**: A professional document summarizing the analysis, findings, and recommendations.

## Directory Structure
- `app/`: Contains the main analysis script (`main.py`).
- `docs/`: Contains the dataset (`crop_yield.csv`).
- `logs/`: Contains log files generated during the analysis.
- `output/`: Contains output files such as CSVs and analysis summaries.
- `plots/`: Contains visualizations generated during the analysis.
- `reports/`: Contains the generated HTML and PDF reports.
- `requirements.txt`: Lists the required Python libraries for the project.

## Installation
To run this project, ensure you have Python installed and then install the required libraries using:

```bash
pip install -r requirements.txt
```

## Usage
Run the analysis script using:

```bash
python app/main.py
```

This will generate the analysis results, visualizations, and reports in the respective directories.
