# ReceiptCountPrediction

## Project Overview
This project aims to predict the number of receipts that will be scanned within a specific duration in 2022. It utilizes a machine learning model to estimate receipt counts based on historical data, trends, and seasonal patterns.

## Files Description

### `modelpy.ipynb`
This Jupyter notebook is the core of the project, divided into three parts:
1. **Data Handling**: Importing data into a DataFrame and extracting the day, week number, and month number. Decomposing the data to analyze Trend, Seasonality, and Residue.
2. **Model Development**: Implementing a linear regression model to fit the weekly trend. The model is trained on 80% of 2021 data and tested on the remaining 20%. It predicts based on the week number of the year.
3. **Sample Testing**: Inputs are the week number in 2022 (ranging from 52+00 to 52+52). The model outputs the average weekly prediction, adjusted for seasonal effects and an offset for standard deviation.

### `app.py`
This Flask application integrates the model with HTML inputs, computes prediction statistics, and generates graphical visualizations.

### `model.pkl`
The saved model from the second part of `modelpy.ipynb`.

### `myhtmlpage.html`, `styles.css`
Contains the HTML structure and styling for the Flask application.

## Setup Instructions
1. Download the repository into a single folder.
2. Ensure the work environment matches the specifications in `requirements.txt`.
3. Run `app.py`. After launching, a local host URL will appear in the terminal. Copy this URL to a browser, input the required data, and click 'Predict'. The outcome, including any error messages, will be displayed.

## Modes of Operation
1. **Pre-Defined Month Selection**: Select a month from a dropdown menu and submit.
2. **Custom Date Range**: Input a custom start and end date in YYYY-MM-DD format.

## Points to Remember
- Input dates in YYYY-MM-DD format only.
- Use either the pre-defined month or custom date inputs, not both.
