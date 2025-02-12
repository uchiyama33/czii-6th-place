import glob
import pandas as pd


def main():
    pattern = "/home/tomo/kaggle/polnet/output/sim_241130/simulated/tomogram_*/labels_table.csv"
    files = glob.glob(pattern)

    if not files:
        print("No files found.")
        return

    dataframes = {}
    for file in files:
        df = pd.read_csv(file)
        dataframes[file] = df

    first_df = next(iter(dataframes.values()))
    all_same = all(df.equals(first_df) for df in dataframes.values())

    if all_same:
        print("All files have the same content.")
    else:
        print("Files with different content:")
        for file, df in dataframes.items():
            if not df.equals(first_df):
                print(file)


if __name__ == "__main__":
    main()
