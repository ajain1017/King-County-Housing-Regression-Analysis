
# Anubhav's King County Housing Regression Analysis


## Introduction

In this project, I explored the cycle of Data Science through the use of the King County Housing Data to help give Home Owners a spot on house price estimate for when they sell.


## Question

Can we accurately predict housing sale price estimates and what are the best features that describe these prices?

Other questions answered:
On what day and in which season were most of these house bought?
Are the different grades located in specific locations?
Are the different group of price ranges 

## Overview of Data Set

The dataset provided was comprised of house sale prices for King County spanning over a year and the houses' descriptive variables. By the end of the project, I had cleaned the data, delt with the null values and missing data, binned and created categorical dummy variables, and feature engineering the rest (through log normalization and minmax scaling).

#### On what day and in which season were most of these house bought?


#### Are the different grades located in specific locations?


#### Are the different group of price ranges



## Exploratory Data Analysis

### Process

Throughout my EDA, I generally went through a simple process to finding my features:

Business Understanding, Data Mining, Data Cleaning, Cleaning and wrangling data, Feature Engineering, Data Exploration, Predictive Modelling

Business Understanding - Clear understanding of problem domain and questions needed to be answered
Overview of Data Cleaning/Feature Engineering - Cleaning and wrangling data, as well as transforming raw data into meaningful features
Questions 1, 2, 3 - Extra questions
Model Results - Data analysis methods to answer question and presenting the findings

#### Feature Engineering:
We use 

## Results

### Summary of Model Statistics

    R-Squared of Train Set:
        0.774

    R-Squared of Test Set:
        0.758

    Train Mean Squared Error:
        0.00420

    Test Mean Squared Error:
        0.00427

    Difference of Mean Squared Errors:
        approximately 0.0

    Mean Absolute Error of Test Set:
        0.0500

    Root Mean Squared Error of Test Set:
        0.0654

    Average Actual Price:
        $591,276

    Average Predicted Price:
        $575,981

    Difference:
        $15,295

### Model Features Selected

After running the feature selection algorithm, we are left with these features with acceptable p-values at a 5% significant level:

#### Dummy Variables:

Grades between 7-13 -> categorical data

Zipcodes from 98021-98121 and 98141-98200

Years built from 1940 onwards

Conditions 4 and 5

Bedrooms 3.5 - 4

Bathrooms 3.25-3.5

2 floors

#### Continous Data:

Longitute and Latitude

Square Foot Lot (Square footage of property)

Square Foot Above (Square footage of house without the basement)

Square Foot Living 15 (Square footage of the houses of the 15 nearest neighbors)

#### Boolean Data:

Basement (Whether there is a basement in the house)

Waterfront (If it is near water)

Renovated (Whether the house was renovated)

### Five Most Descriptuve Features Using Recursive Feature Elimination

After making the model, we use recursive feature elimination from sklearn Linear Regression to select the highest ranked features from the features selected above in estimating housing prices.

We find that the most influential 