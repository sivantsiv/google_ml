# @title Setup - Import relevant modules
import pandas as pd

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

def find_outliers_iqr(data):
    """
    Find outliers in a numerical dataset using the IQR method.

    Parameters:
        data (list, pd.Series): A 1D list or pandas Series of numeric values.

    Returns:
        outliers (list): List of outlier values.
        bounds (tuple): Lower and upper bounds used for detecting outliers.
    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data < lower_bound) | (data > upper_bound)]

    return outliers.tolist(), (lower_bound, upper_bound)

def main():
    """Main is entry point for the script."""

    # Read the dataset from the URL.    
    training_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

    # The following code returns basic statistics about the data in the dataframe.
    print("The basic data statistics are:", training_df.describe())

    # Generalized outlier detection for all numeric columns
    for col in training_df.select_dtypes('number').columns:
        outliers, bounds = find_outliers_iqr(training_df[col])
        print(f"{col} outliers: {len(outliers)} found, bounds: {bounds}")

if __name__ == "__main__":
    main()
