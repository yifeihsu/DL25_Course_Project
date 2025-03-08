import pandas as pd


def compare_submissions(file1, file2):
    # Read both CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Check if they have the same shape (same number of rows and columns)
    if df1.shape != df2.shape:
        print(f"Files '{file1}' and '{file2}' have different shapes.")
        print(f"{file1} shape: {df1.shape}")
        print(f"{file2} shape: {df2.shape}")
        return

    # Compare entire DataFrames
    differences = df1.compare(df2)

    if differences.empty:
        print(f"The files '{file1}' and '{file2}' are identical.")
    else:
        print(f"The files '{file1}' and '{file2}' differ. Below are the mismatched rows/columns:")
        print(differences.head(10))  # Print just the first 10 mismatches for brevity


if __name__ == "__main__":
    compare_submissions("submission1.csv", "submission_c.csv")