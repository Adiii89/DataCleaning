from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import base64
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd



def main():
    # Title of the web app
    st.title("DataMorph: Transforming Data with Preprocessing, EDA, and Visualizations")
    app_mode = st.sidebar.selectbox('Choose the App Mode', ['About Website', 'EDA', 'Plot', 'Data Preprocessing'])
    if app_mode== 'About Website':
        st.header('Introduction ', divider='rainbow')
         with st.container():

           st.write('This web application serves as a tool for Exploratory Data Analysis (EDA), Data Visualization, and Data Preprocessing. It is built using Streamlit, a Python library for creating web applications for machine learning and data science projects. The application allows users to upload datasets in CSV format and perform different tasks such as visualizing data, exploring summary statistics, handling missing values, outliers, and more.')
        st.header('Components of the Application:', divider='rainbow')
        st.write(" **1.Exploratory Data Analysis (EDA):** "
                  "Users can upload a dataset and explore its contents. They can view the first few rows of the dataset, summary statistics, column names, and value counts of selected columns. Options are available to visualize data using Matplotlib and Seaborn for correlation plots, and to generate pie charts.")
         st.write(" **Data Visualization:** "
                  "Users can upload a dataset and visualize it using different types of plots such as area, bar, line, histogram, box, and kernel density estimate (kde). They can select specific columns to plot and customize the visualization according to their preferences.")
         st.write(" **Data Preprocessing:** "
                  "Users can upload a dataset and choose various preprocessing tasks such as removing duplicates, filling missing values, handling categorical data, normalization, standardization, and handling outliers. Options are available for different strategies to handle missing values and outliers. After preprocessing, users can download the cleaned dataset in CSV format.")
         st.header('Conclusion', divider='rainbow')
         st.write("This web application provides a user-friendly interface for data exploration, visualization, and preprocessing tasks. It empowers users to gain insights from their datasets, prepare data for machine learning models, and streamline their data analysis workflows. With its intuitive design and comprehensive functionality, it serves as a valuable tool for data scientists, analysts, and researchers in their data-driven endeavors.")

        
    elif app_mode == 'EDA':
        st.title('Exploratory Data Analysis')
        data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            all_columns = df.columns.to_list()
            column_to_plot = st.selectbox("Select 1 Column", all_columns)

            if st.checkbox("Show Shape"):
                st.write(df.shape)

            if st.checkbox("Show Columns"):
                st.write(all_columns)

            if st.checkbox("Summary"):
                st.write(df.describe())

            if st.checkbox("Show Selected Columns"):
                selected_columns = st.multiselect("Select Columns", all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)

            if st.checkbox("Show Value Counts"):
                st.write(df[column_to_plot].value_counts())

            if st.checkbox("Correlation Plot(Matplotlib)"):
                plt.matshow(new_df.corr())
                st.pyplot()

            if st.checkbox("Correlation Plot(Seaborn)"):
                correlatio_plot = sns.heatmap(new_df.corr(method='pearson', min_periods=1), annot=True)
                st.write(correlatio_plot)
                st.pyplot()

            if st.checkbox("Pie Plot"):
                pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

    elif app_mode == 'Plot':
        st.title('Data Visualization')
        data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:, -1].value_counts().plot(kind='bar'))
                st.pyplot()

            # Customizable Plot

            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "box", "kde"])
            selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)

            if st.button("Generate Plot"):
                st.success("Generating Customizable Plot of {} for {}".format(type_of_plot, selected_columns_names))

                # Plot By Streamlit
                if type_of_plot == 'area':
                    cust_data = df[selected_columns_names]
                    st.area_chart(cust_data)

                elif type_of_plot == 'bar':
                    cust_data = df[selected_columns_names]
                    st.bar_chart(cust_data)

                elif type_of_plot == 'line':
                    cust_data = df[selected_columns_names]
                    st.line_chart(cust_data)

                # Custom Plot
                elif type_of_plot:
                    cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
                    st.write(cust_plot)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()

    elif app_mode == 'Data Preprocessing':

        st.title('Data Cleaning App')

        def clean_data(data, remove_duplicates, fill_missing_values, ways_of_filling_missing_value, normalization,
                       standardization, handle_categorical_data, handle_outliers):
            if remove_duplicates:
                data = data.drop_duplicates()
            if fill_missing_values:
                if ways_of_filling_missing_value == 'Drop Null values':
                    x = (70 * len(data)) / 100
                    y = (70 * len(data.columns)) / 100
                    threshold = len(data) - x
                    threshold2 = len(data.columns) - y  # Keep columns with less than x null values

                    data = data.dropna(axis=1, thresh=threshold)  # Drop columns with mostly null values
                    data = data.dropna(axis=0, thresh=threshold2)  # Drop rows with mostly null values
                    data = data.dropna()
                elif ways_of_filling_missing_value == 'Replace Null Values with Mean':
                    x = (70 * len(data)) / 100
                    y = (70 * len(data.columns)) / 100
                    threshold = len(data) - x
                    threshold2 = len(data.columns) - y  # Keep columns with less than x null values

                    data = data.dropna(axis=1, thresh=threshold)  # Drop columns with mostly null values
                    data = data.dropna(axis=0, thresh=threshold2)  # Drop rows with mostly null values

                    def replace_missing_with_mean_mode(df):
                        for column in df.columns:
                            if df[column].dtype == 'object':  # Check if column is categorical
                                mode = df[column].mode()[0]  # Compute the mode for the column
                                df[column].fillna(mode, inplace=True)  # Replace missing values with the mode
                            elif df[column].dtype in ['int64', 'float64']:  # Check if column is numerical
                                mean = df[column].mean()  # Compute the mean for the column
                                df[column].fillna(mean, inplace=True)  # Replace missing values with the mean
                        return data

                    data = replace_missing_with_mean_mode(data)
                    # data = data.fillna(0)  # You can replace 0 with any value you want to fill missing values with
            if handle_categorical_data:
                # Create label encoder object
                label_encoder = preprocessing.LabelEncoder()

                # Iterate over each column in the DataFrame
                for column in data.columns:
                    # Check if the column is of object type (categorical)
                    if data[column].dtype == 'object':
                        # Encode labels in the column
                        data[column] = label_encoder.fit_transform(data[column].astype(str))

                # Display the processed DataFrame
                print("Processed DataFrame:")

            if normalization:
                numeric_columns = data.select_dtypes(include=['int', 'float']).columns

                # Normalize the numeric columns using Min-Max scaling
                scaler = MinMaxScaler()
                data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

                # Print the DataFrame after normalization
                print("DataFrame after normalization:")

            if standardization:
                # Select only numeric columns for standardization
                numeric_columns = data.select_dtypes(include=['int', 'float']).columns

                # Standardize the numeric columns
                scaler = StandardScaler()
                data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

                # Print the DataFrame after standardization
                print("DataFrame after standardization:")

            if handle_outliers:
                if handle_outliers == 'Drop':
                    iqr_multiplier = 1.5

                    # Function to handle outliers for all columns using IQR methoda
                    def handle_outliers_iqr(data, iqr_multiplier):
                        for column in data.columns:
                            # Check if the column contains numeric data
                            if pd.api.types.is_numeric_dtype(data[column]):
                                # Calculate the first and third quartiles
                                Q1 = data[column].quantile(0.25)
                                Q3 = data[column].quantile(0.75)
                                # Calculate the IQR
                                IQR = Q3 - Q1
                                # Define the lower and upper bounds for outliers
                                lower_bound = Q1 - iqr_multiplier * IQR
                                upper_bound = Q3 + iqr_multiplier * IQR
                                # Identify outliers
                                outlier_indices = (data[column] < lower_bound) | (data[column] > upper_bound)
                                # Replace outliers with NaN
                                # data.loc[outlier_indices, column] = None  # or any value you prefer for replacing outliers
                                data = data[~outlier_indices]
                        return data

                    # Handle outliers for all columns using IQR method
                    data = handle_outliers_iqr(data, iqr_multiplier)

                elif handle_outliers == 'Impute Null':
                    iqr_multiplier = 1.5

                    # Function to handle outliers for all columns using IQR methoda
                    def handle_outliers_iqr(data, iqr_multiplier):
                        for column in data.columns:
                            # Check if the column contains numeric data
                            if pd.api.types.is_numeric_dtype(data[column]):
                                # Calculate the first and third quartiles
                                Q1 = data[column].quantile(0.25)
                                Q3 = data[column].quantile(0.75)
                                # Calculate the IQR
                                IQR = Q3 - Q1
                                # Define the lower and upper bounds for outliers
                                lower_bound = Q1 - iqr_multiplier * IQR
                                upper_bound = Q3 + iqr_multiplier * IQR
                                # Identify outliers
                                outlier_indices = (data[column] < lower_bound) | (data[column] > upper_bound)
                                # Replace outliers with NaN
                                data.loc[
                                    outlier_indices, column] = None  # or any value you prefer for replacing outliers
                                # data = data[~outlier_indices]
                        return data

                    # Handle outliers for all columns using IQR method
                    data = handle_outliers_iqr(data, iqr_multiplier)
                elif handle_outliers == 'Impute Mean':
                    iqr_multiplier = 1.5

                    # Function to handle outliers for all columns using IQR methoda
                    def handle_outliers_iqr(data, iqr_multiplier):
                        for column in data.columns:
                            # Check if the column contains numeric data
                            if pd.api.types.is_numeric_dtype(data[column]):
                                # Calculate the first and third quartiles
                                Q1 = data[column].quantile(0.25)
                                Q3 = data[column].quantile(0.75)
                                # Calculate the IQR
                                IQR = Q3 - Q1
                                # Define the lower and upper bounds for outliers
                                lower_bound = Q1 - iqr_multiplier * IQR
                                upper_bound = Q3 + iqr_multiplier * IQR
                                # Identify outliers
                                outlier_indices = (data[column] < lower_bound) | (data[column] > upper_bound)
                                # Replace outliers with NaN
                                # data.loc[outlier_indices, column] = None  # or any value you prefer for replacing outliers
                                # data = data[~outlier_indices]
                                data.loc[outlier_indices, column] = data[column].mean()
                        return data

                    # Handle outliers for all columns using IQR method
                    data = handle_outliers_iqr(data, iqr_multiplier)
                elif handle_outliers == 'Impute Mode':
                    iqr_multiplier = 1.5

                    # Function to handle outliers for all columns using IQR method
                    def handle_outliers_iqr(data, iqr_multiplier):
                        for column in data.columns:
                            # Check if the column contains numeric data
                            if pd.api.types.is_numeric_dtype(data[column]):
                                # Calculate the first and third quartiles
                                Q1 = data[column].quantile(0.25)
                                Q3 = data[column].quantile(0.75)
                                # Calculate the IQR
                                IQR = Q3 - Q1
                                # Define the lower and upper bounds for outliers
                                lower_bound = Q1 - iqr_multiplier * IQR
                                upper_bound = Q3 + iqr_multiplier * IQR
                                # Identify outliers
                                outlier_indices = (data[column] < lower_bound) | (data[column] > upper_bound)
                                # Impute outliers with mode
                                data.loc[outlier_indices, column] = data[column].mode().iloc[
                                    0]  # Take the first mode if multiple exist
                        return data

                    # Handle outliers for all columns using IQR method and impute mode
                    data = handle_outliers_iqr(data, iqr_multiplier)
                elif handle_outliers == 'Impute Median':
                    iqr_multiplier = 1.5

                    # Function to handle outliers for all columns using IQR method
                    def handle_outliers_iqr(data, iqr_multiplier):
                        for column in data.columns:
                            # Check if the column contains numeric data
                            if pd.api.types.is_numeric_dtype(data[column]):
                                # Calculate the first and third quartiles
                                Q1 = data[column].quantile(0.25)
                                Q3 = data[column].quantile(0.75)
                                # Calculate the IQR
                                IQR = Q3 - Q1
                                # Define the lower and upper bounds for outliers
                                lower_bound = Q1 - iqr_multiplier * IQR
                                upper_bound = Q3 + iqr_multiplier * IQR
                                # Identify outliers
                                outlier_indices = (data[column] < lower_bound) | (data[column] > upper_bound)
                                # Impute outliers with median
                                data.loc[outlier_indices, column] = data[column].median()
                        return data

                    # Handle outliers for all columns using IQR method and impute median
                    data = handle_outliers_iqr(data, iqr_multiplier)

            return data

        def download_csv(cleaned_data):
            csv = cleaned_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download CSV File</a>'
            return href

        # Upload CSV file
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

        if uploaded_file is not None:
            # Read the uploaded CSV file
            data = pd.read_csv(uploaded_file)

            # Display the raw data
            st.subheader('Raw Data')
            st.write(data)
            st.sidebar.header('Select Data Preprocessing parameters')
            # Checkbox for removing duplicates
            remove_duplicates = st.sidebar.checkbox('Remove Duplicates')

            # Checkbox for filling missing values
            fill_missing_values = st.sidebar.checkbox('Fill Missing Values')
            handle_categorical_data = st.sidebar.checkbox('Handle Categorical Values')
            normalization = st.sidebar.checkbox('Normalization')
            standardization = st.sidebar.checkbox('Standardization')
            handle_outliers = st.sidebar.checkbox('Handle Outliers')

            if fill_missing_values:
                Handling_missing_value = st.sidebar.selectbox('Select the way you want to handle the Null Values', ['Drop Null values', 'Replace Null Values with Mean'])
            if handle_outliers:
                # Sub-selectbox for handling outliers
                outlier_strategy = st.sidebar.selectbox('Select Outlier Handling Strategy', ['Drop', 'Impute Null', 'Impute Mean', 'Impute Mode', 'Impute Median'])


            # Button to trigger data cleaning
            if st.sidebar.button('Clean Data'):
                if fill_missing_values:
                    cleaned_data = clean_data(data, remove_duplicates, fill_missing_values, Handling_missing_value, normalization, standardization, handle_categorical_data,handle_outliers)
                else:
                    cleaned_data = clean_data(data, remove_duplicates, fill_missing_values,  None, normalization, standardization, handle_categorical_data,handle_outliers)

                # Display the cleaned data
                st.subheader('Cleaned Data')
                st.write(cleaned_data)

                if cleaned_data is not None:
                    # Download button for cleaned data
                    st.markdown(download_csv(cleaned_data), unsafe_allow_html=True)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
