# ReceiptCountPrediction

## Project Overview
This project aims to predict the number of receipts that will be scanned within a specific duration in 2022. It utilizes a machine learning model to estimate receipt counts based on historical data, trends, and seasonal patterns.

## Files Description

### `modelpy.py`
This Jupyter notebook is the core of the project and is divided into three parts:
1. **Data Handling**: Importing data into a DataFrame and extracting the day, week, and month numbers. Decomposing the data to analyze Trend, Seasonality, and Residue.
2. **Model Development**: Implementing a simple Linear Regression model but finding the coefficients(intercept, slope) by finding the closed form solution. Using a dot product with the coefficients, we can predict the values.
3. **Sample Testing**: Inputs are the week number in 2022 (ranging from 52+00 to 52+52). The model outputs the average weekly prediction, adjusted for seasonal effects and an offset for standard deviation.

### `app.py`
This Flask application integrates the model with HTML inputs, computes prediction statistics, and generates graphical visualizations.


### `myhtmlpage.html`, `styles.css`
It contains the HTML structure and styling for the Flask application.

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
- Docker is not initialized yet,i am facing some errors in imaging my file,avoid the docker files
