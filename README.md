# DataMorph-Preprocessing_and_EDA Web-app Tool

## Index

- [Introduction](#introduction)

- [Demo](#demo)

- [Features](#features)

- [Implementation Details](#implementation-details)

- [Installation](#installation)

- [Contributors](#Contributors)


## Introduction
This web application serves as a tool for Exploratory Data Analysis (EDA), Data Visualization, and Data Preprocessing. It is built using Streamlit, a Python library for creating web applications for machine learning and data science projects. The application allows users to upload datasets in CSV format and perform different tasks such as visualizing data, exploring summary statistics, handling missing values, outliers, and more.

<a name="introduction"></a>

## Demo

  https://adiii89-datacleaning-streamlitdatacleaning-gmtrad.streamlit.app/

<a name="demo"></a>

## Features

#### Exploratory Data Analysis (EDA):

 -Upload a dataset and explore its contents.

 -View the first few rows of the dataset, summary statistics, column names, and value counts of selected columns.

 -Visualize data using Matplotlib and Seaborn for correlation plots, and generate pie charts.

#### Data Visualization:
 
 -Upload a dataset and visualize it using different types of plots such as area, bar, line, histogram, box, and kernel density estimate (kde).

  -Select specific columns to plot and customize the visualization according to preferences.

#### Data Preprocessing:

  -Upload a dataset and choose various preprocessing tasks such as removing duplicates, filling missing values, handling categorical data, normalization, standardization, and handling outliers.

  -Choose different strategies to handle missing values and outliers.
  
  -Download the cleaned dataset in CSV format after preprocessing.

<a name="features"></a>

## Implementation Details

  -Organized into different modes selectable from the sidebar: "About Website," "EDA," "Plot," and "Data Preprocessing."
  
  -Each mode provides specific functionalities tailored to the corresponding task.
  
  -Libraries such as Pandas, Seaborn, Matplotlib, and Scikit-learn (for preprocessing) are used for data manipulation, visualization, and preprocessing.
  
  -The application is structured using functions, with separate functions for each mode of operation and specific tasks within each mode.
  
  -Error handling is implemented to ensure smooth execution and a user-friendly experience.

<a name="implementation-details"></a>

  ## Installation
  
  Install the required dependencies:
  
  pandas~=2.2.1
  
  streamlit~=1.32.2
  
  scikit-learn~=1.4.1.post1
  
  seaborn~=0.13.2
  
  matplotlib
  
  numpy~=1.26.4

<a name="installation"></a>

  ## Contributors
  
  [Akshata Jadhav](https://github.com/Akshata196)
  
  [Aditya Mallesh](https://github.com/Adiii89)
  
<a name="Contributors"></a>





