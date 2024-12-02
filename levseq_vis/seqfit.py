"""
A script for visualzing the sequnece-fitness relationship
"""

from typing import Optional, Dict

import re
import os

import warnings
from copy import deepcopy


import panel as pn
import holoviews as hv
import ninetysix as ns



# Get them w.r.t to a mutation
from scipy.stats import mannwhitneyu
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict

# Enable Bokeh to display plots in the notebook

# Amino acid code conversion
AA_DICT = {
    "Ala": "A",
    "Cys": "C",
    "Asp": "D",
    "Glu": "E",
    "Phe": "F",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Lys": "K",
    "Leu": "L",
    "Met": "M",
    "Asn": "N",
    "Pro": "P",
    "Gln": "Q",
    "Arg": "R",
    "Ser": "S",
    "Thr": "T",
    "Val": "V",
    "Trp": "W",
    "Tyr": "Y",
    "Ter": "*",
}

def calculate_mutation_combinations(stats_df):
    mutation_dict = defaultdict(list)
    for mutation in stats_df['mutation'].values:
        mutations = mutation.split('_')
        for m in mutations:
            mutation_dict[m].append(mutation)

    rows = []
    with pd.ExcelWriter('mutations.xlsx', engine='xlsxwriter') as writer:
        for mutation, mutations in mutation_dict.items():
            # Here we want to now get the values for each of these i.e. the stats values for each one and summarize it maybe for now we'll just make a excel file
            df1 = stats_df[stats_df['mutation'].isin(mutations)]
            mutation = mutation.replace('*', '.')
            df1.to_excel(writer, sheet_name=mutation)
            # Also just take the mean of the mean lol and the sum of the number of the wells
            rows.append([mutation, np.sum(df1['number of wells with mutation'].values), '|'.join(set(list(mutations))),
                         np.mean(df1['mean'].values),
                         np.median(df1['median'].values), np.mean(df1['amount greater than parent mean'].values),
                         np.max(df1['amount greater than parent mean'].values)])

    df = pd.DataFrame(rows, columns=['mutation', 'number of wells with mutation',
                                     'other-mutations', 'mean', 'median',
                                     'mean amount greater than parent', 'max amount greater than parent'])
    df.sort_values(by='mean amount greater than parent', ascending=False)
    return df


def normalise_calculate_stats(processed_plate_df, value_columns, normalise='standard', stats_method='mannwhitneyu',
                              parent_label='#PARENT#', normalise_method='median'):
    parent = parent_label
    # if nomrliase normalize with standard normalisation
    normalised_value_columns = []
    normalised_df = pd.DataFrame()
    if normalise:
        for plate in set(processed_plate_df['Plate'].values):
            for value_column in value_columns:
                sub_df = processed_plate_df[processed_plate_df['Plate'] == plate]
                parent_values = sub_df[sub_df['amino-acid_substitutions'] == parent][value_column].values
                # By default use the median
                if normalise_method == 'median':
                    parent_mean = np.median(parent_values)
                else:
                    parent_mean = np.mean(parent_values)
                parent_sd = np.std(parent_values)

                # For each plate we normalise to the parent of that plate
                sub_df[f'{value_column} plate standard norm'] = (sub_df[value_column].values - parent_mean) / parent_sd
                normalised_value_columns.append(f'{value_column} plate standard norm')
                normalised_df = pd.concat([normalised_df, sub_df])
    else:
        normalised_df = processed_plate_df

    normalised_value_columns = list(set(normalised_value_columns))
    processed_plate_df = normalised_df

    sd_cutoff = 1.5  # The number of standard deviations we want above the parent values
    # Now for all the other mutations we want to look if they are significant, first we'll look at combinations and then individually
    grouped_by_mutations = processed_plate_df.groupby('amino-acid_substitutions')

    rows = []
    for mutation, grp in tqdm(grouped_by_mutations):
        # Get the values and then do a ranksum test
        if mutation != parent:
            for value_column in normalised_value_columns:
                parent_values = list(processed_plate_df[processed_plate_df['amino-acid_substitutions'] == parent][value_column].values)
                if normalise_method == 'median':
                    parent_mean = np.median(parent_values)
                else:
                    parent_mean = np.mean(parent_values)
                parent_sd = np.std(parent_values)

                vals = list(grp[value_column].values)
                U1, p = None, None
                # Now check if there are 3 otherwise we just do > X S.D over - won't be sig anyway.
                if len(grp) > 2:
                    # Do stats
                    U1, p = mannwhitneyu(parent_values, vals, method="exact")
                mean_vals = np.mean(vals)
                std_vals = np.std(vals)
                median_vals = np.median(vals)
                sig = mean_vals > ((sd_cutoff * parent_sd) + parent_mean)
                rows.append(
                    [value_column, mutation, len(grp), mean_vals, std_vals, median_vals, mean_vals - parent_mean, sig,
                     U1, p])
    stats_df = pd.DataFrame(rows, columns=['value_column', 'amino-acid_substitutions', 'number of wells with amino-acid substitutions', 'mean', 'std',
                                           'median', 'amount greater than parent mean',
                                           f'greater than > {sd_cutoff} parent', 'man whitney U stat', 'p-value'])
    return stats_df


def checkNgen_folder(folder_path: str) -> str:

    """
    Check if the folder and its subfolder exists
    create a new directory if not
    Args:
    - folder_path: str, the folder path
    """
    # get rid of the very first / if it exists
    if folder_path[0] == "/":
        folder_path = folder_path[1:]

    # if input path is file
    if bool(os.path.splitext(folder_path)[1]):
        folder_path = os.path.dirname(folder_path)

    split_list = os.path.normpath(folder_path).split("/")
    for p, _ in enumerate(split_list):
        subfolder_path = "/".join(split_list[: p + 1])
        if not os.path.exists(subfolder_path):
            print(f"Making {subfolder_path} ...")
            os.mkdir(subfolder_path)
    return folder_path




def work_up_lcms(
    file,
    products,
    substrates=None,
    drop_string=None,
):
    """Works up a standard csv file from Revali.
    Parameters:
    -----------
    file: string
        Path to the csv file
    products: list of strings
        Name of the peaks that correspond to the product
    substrates: list of strings
        Name of the peaks that correspond to the substrate
    drop_string: string, default 'burn_in'
        Name of the wells to drop, e.g., for the wash/burn-in period that are not samples.
    Returns:
    --------
    plate: ns.Plate object (DataFrame-like)
    """
    if isinstance(file, str):
        # Read in the data
        df = pd.read_csv(file, header=[1])
    else:
        # Change to handling both
        df = file
    # Convert nans to 0
    df = df.fillna(0)
    # Only grab the Sample Acq Order No.s that have a numeric value
    index = [True for _ in df["Sample Acq Order No"]]
    for i, value in enumerate(df["Sample Acq Order No"]):
        try:
            int(value)
        except ValueError:
            index[i] = False
    # Index on this
    df = df[index]

    def fill_vial_number(series):
        for i, row in enumerate(series):
            if pd.isna(row):
                series[i] = series[i - 1]
        return series

    df["Sample Vial Number"] = fill_vial_number(df["Sample Vial Number"].copy())
    # Drop empty ones!
    df = df[df["Sample Vial Number"] != 0]
    # Remove unwanted wells
    df = df[df["Sample Name"] != drop_string]
    # Get wells

    df.insert(0, "Well", df["Sample Vial Number"].apply(lambda x: str(x).split("-")[-1]))
    # Rename
    df = df.rename({"Sample Name": "Plate"}, axis="columns")
    # Create minimal DataFrame
    df = df[["Well", "Plate", "Compound Name", "Area"]].reset_index(drop=True)
    # Pivot table; drop redundant values by only taking 'max' with aggfunc
    # (i.e., a row is (value, NaN, NaN) and df is 1728 rows long;
    # taking max to aggregate duplicates gives only (value) and 576 rows long)
    df = df.pivot_table(
        index=["Well", "Plate"], columns="Compound Name", values="Area", aggfunc="max"
    ).reset_index()
    # Get rows and columns
    df.insert(1, "Column", df["Well"].apply(lambda x: int(x[1:]) if x[1:].isdigit() else None))
    df.insert(1, "Row", df["Well"].apply(lambda x: x[0]))
    # Set values as floats
    cols = products + substrates if substrates is not None else products
    for col in cols:
        df[col] = df[col].astype(float)
    plate = ns.Plate(df, value_name=products[-1]).set_as_location("Plate", idx=3)
    plate.values = products
    return plate


def process_files(results_df, plate_df, plate: str, product: list) -> pd.DataFrame:
    """
    Process and combine a single plate file

    Args:
    - product : str
        The name of the product to be analyzed. ie pdt
    - plate : str, ie 'HMC0225_HMC0226.csv'
        The name of the input CSV file containing the plate data.

    Returns:
    - pd.DataFrame
        A pandas DataFrame containing the processed data.
    - str
        The path of the output CSV file containing the processed data.
    """
    filtered_df = results_df[["Plate", "Well", "amino-acid_substitutions", "nt_sequence", "aa_sequence"]]
    filtered_df = filtered_df[(filtered_df["amino-acid_substitutions"] != "#N.A.#")].dropna()

    # Extract the unique entries of Plate
    unique_plates = filtered_df["Plate"].unique()

    # Create an empty list to store the processed plate data
    processed_data = []

    # Iterate over unique Plates and search for corresponding CSV files in the current directory
    plate_object = work_up_lcms(plate_df, product)

    # Extract attributes from plate_object as needed for downstream processes
    if hasattr(plate_object, "df"):
        # Assuming plate_object has a dataframe-like attribute 'df' that we can work with
        plate_df = plate_object.df
        plate_df["Plate"] = plate  # Add the plate identifier for reference

        # Merge filtered_df with plate_df to retain amino-acid_substitutionss and nt_sequence columns
        merged_df = pd.merge(
            plate_df, filtered_df, on=["Plate", "Well"], how="left"
        )
        columns_order = (
                ["Plate", "Well", "Row", "Column", "amino-acid_substitutions"]
                + product
                + ["nt_sequence", "aa_sequence"]
        )
        merged_df = merged_df[columns_order]
        processed_data.append(merged_df)

    # Concatenate all dataframes if available
    if processed_data:
        processed_df = pd.concat(processed_data, ignore_index=True)
    else:
        processed_df = pd.DataFrame(
            columns=["Plate", "Well", "Row", "Column", "amino-acid_substitutions"]
                    + product
                    + ["nt_sequence", "aa_sequence"]
        )

    # Ensure all entries in 'Mutations' are treated as strings
    processed_df["amino-acid_substitutions"] = processed_df["amino-acid_substitutions"].astype(str)

    # Remove any rows with empty values
    processed_df = processed_df.dropna()

    # Return the processed DataFrame for downstream processes
    return processed_df


# Function to process the plate files
def process_plate_files(product: str, input_csv: str) -> pd.DataFrame:

    """
    Process the plate files to extract relevant data for downstream analysis.
    Assume the same directory contains the plate files with the expected names.
    The expected filenames are constructed based on the Plate values in the input CSV file.
    The output DataFrame contains the processed data for the specified product
    and is saved to a CSV file named 'seqfit.csv' in the same dirctory.

    Args:
    - product : str
        The name of the product to be analyzed. ie pdt
    - input_csv : str, ie 'HMC0225_HMC0226.csv'
        The name of the input CSV file containing the plate data.

    Returns:
    - pd.DataFrame
        A pandas DataFrame containing the processed data.
    - str
        The path of the output CSV file containing the processed data.
    """

    dir_path = os.path.dirname(input_csv)
    print(f"Processing data from '{dir_path}'")

    # Load the provided CSV file
    results_df = pd.read_csv(input_csv)

    # Extract the required columns: Plate, Well, amino-acid_substitutionss, and nt_sequence, and remove rows with '#N.A.#' and NaN values
    # barcode_plate	Plate	Well	Alignment Count	nucleotide_amino-acid_substitutions	amino-acid_substitutions	Alignment Probability	Average amino-acid_substitutions frequency	P value	P adj. value	nt_sequence	aa_sequence
    filtered_df = results_df[["Plate", "Well", "amino-acid_substitutions", "nt_sequence", "aa_sequence"]]
    filtered_df = filtered_df[(filtered_df["amino-acid_substitutions"] != "#N.A.#")].dropna()

    # Extract the unique entries of Plate
    unique_plates = filtered_df["Plate"].unique()

    # Create an empty list to store the processed plate data
    processed_data = []

    # Iterate over unique Plates and search for corresponding CSV files in the current directory
    for plate in unique_plates:
        # Construct the expected filename based on the Plate value
        filename = os.path.join(dir_path, f"{plate}.csv")

        # Check if the file exists in the current directory
        if os.path.isfile(filename):
            print(f"Processing data for Plate: {plate}")
            # Work up data to plate object
            plate_object = work_up_lcms(filename, product)

            # Extract attributes from plate_object as needed for downstream processes
            if hasattr(plate_object, "df"):
                # Assuming plate_object has a dataframe-like attribute 'df' that we can work with
                plate_df = plate_object.df
                plate_df["Plate"] = plate  # Add the plate identifier for reference

                # Merge filtered_df with plate_df to retain amino-acid_substitutionss and nt_sequence columns
                merged_df = pd.merge(
                    plate_df, filtered_df, on=["Plate", "Well"], how="left"
                )
                columns_order = (
                    ["Plate", "Well", "Row", "Column", "amino-acid_substitutions"]
                    + product
                    + ["nt_sequence", "aa_sequence"]
                )
                merged_df = merged_df[columns_order]
                processed_data.append(merged_df)

    # Concatenate all dataframes if available
    if processed_data:
        processed_df = pd.concat(processed_data, ignore_index=True)
    else:
        processed_df = pd.DataFrame(
            columns=["Plate", "Well", "Row", "Column", "amino-acid_substitutions"]
            + product
            + ["nt_sequence", "aa_sequence"]
        )

    # Ensure all entries in 'Mutations' are treated as strings
    processed_df["amino-acid_substitutions"] = processed_df["amino-acid_substitutions"].astype(str)

    # Remove any rows with empty values
    processed_df = processed_df.dropna()

    seqfit_path = os.path.join(dir_path, "seqfit.csv")

    # Optionally, save the processed DataFrame to a CSV file
    processed_df.to_csv(seqfit_path, index=False)
    print(f"Processed data saved to {seqfit_path} in the same directory")

    # Return the processed DataFrame for downstream processes
    return processed_df, seqfit_path


def detect_outliers_iqr(series: pd.Series) -> pd.Index:

    """
    Calculate the Interquartile Range (IQR) and
    determine the lower and upper bounds for outlier detection.

    The IQR is a measure of statistical dispersion and
    is calculated as the difference between the third quartile (Q3)
    and the first quartile (Q1) of the data

    Args:
    - series : pandas.Series
        A pandas Series containing the data for which the IQR and bounds are to be calculated.

    Returns:
    - tuple
        A tuple containing the lower bound and upper bound for outlier detection.

    Example:
    --------
    >>> import pandas as pd
    >>> data = pd.Series([10, 12, 14, 15, 18, 20, 22, 23, 24, 25, 100])
    >>> calculate_iqr_bounds(data)
    (-1.0, 39.0)
    """

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return series[(series < lower_bound) | (series > upper_bound)].index


def norm2parent(plate_df: pd.DataFrame) -> pd.DataFrame:

    """
    For each given plate,
    normalize the pdt values of a plate to the mean of the parent
    without the outliers.

    Args:
    - plate_df : pd.DataFrame
        A pandas DataFrame containing the data for a single plate.
        The DataFrame should have the following columns:
        - "Plate" : str
            The plate identifier.
        - "Mutations" : str
            The mutations in the well.
        - "pdt" : float
            The pdt value for the well.

    Returns:
    - pd.DataFrame
        A pandas DataFrame containing the normalized pdt values.
    """

    # get all the parents from the df
    parents = (
        plate_df[plate_df["amino-acid_substitutions"] == "#PARENT#"].reset_index(drop=True).copy()
    )
    filtered_parents = (
        parents.drop(index=detect_outliers_iqr(parents["pdt"]))
        .reset_index(drop=True)
        .copy()
    )

    # normalize the whole plate to the mean of the filtered parent
    plate_df["pdt_norm"] = plate_df["pdt"] / filtered_parents["pdt"].mean()

    return plate_df


def process_mutation(mutation: str) -> pd.Series:
    # Check if mutation is #PARENT#
    if mutation == "#PARENT#":
        return pd.Series([0, [(None, None, None)]])  # Return 0 sites and NaN details

    # Split by "_" to get number of sites
    sites = mutation.split("_")
    num_sites = len(sites)

    # Extract details if it matches the pattern
    details = []
    for site in sites:
        match = re.match(r"^([A-Z])(\d+)([A-Z*])$", site)
        if match:
            parent_aa, site_number, mutated_aa = match.groups()
            details.append((parent_aa, site_number, mutated_aa))
        else:
            details.append((None, None, None))

    return pd.Series([num_sites, details])


def prep_single_ssm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the data for a single sitessm summary plot.

    Args:
    - df: pd.DataFrame, input full dataframe

    Returns:
    - pd.DataFrame, output dataframe
    """

    # slice out single site SSM and add in parentAA, site, and mutAA columns
    single_ssm_df = df[df["num_sites"] <= 1].copy()

    # Expand the single entry in Details for these rows into three columns
    single_ssm_df[["parent_aa", "site_numb", "mut_aa"]] = pd.DataFrame(
        single_ssm_df["mut_dets"].apply(lambda x: x[0]).tolist(),
        index=single_ssm_df.index,
    )

    single_ssm_df["parent_aa_loc"] = (
        single_ssm_df["parent_aa"] + single_ssm_df["site_numb"]
    )

    # fill nan site numbers with 0 and convert to int
    single_ssm_df["site_numb"] = single_ssm_df["site_numb"].fillna(0).astype(int)

    return single_ssm_df


def get_single_ssm_site_df(
    single_ssm_df: pd.DataFrame, parent: str, site: str
) -> pd.DataFrame:
    """
    Get the single site SSM data for a given site with appended parent data.

    Args:
    - single_ssm_df: pd.DataFrame, input single site SSM dataframe
    - parent: str, parent to filter the data on
    - site: str, site to filter the data on

    Returns:
    - pd.DataFrame, output dataframe
    """

    # get the site data
    site_df = (
        single_ssm_df[
            (single_ssm_df["Parent_Name"] == parent)
            & (single_ssm_df["parent_aa_loc"] == site)
        ]
        .reset_index(drop=True)
        .copy()
    )

    # get parents from those plates
    site_parent_df = (
        single_ssm_df[
            (single_ssm_df["amino-acid_substitutions"] == "#PARENT#")
            & (single_ssm_df["Plate"].isin(site_df["Plate"].unique()))
        ]
        .reset_index(drop=True)
        .copy()
    )

    # rename those site_numb, mut_aa, parent_aa_loc None or NaN to corresponding parent values
    site_parent_df["mut_aa"] = site_parent_df["mut_aa"].fillna(
        site_df["parent_aa"].values[0]
    )
    site_parent_df["site_numb"] = site_parent_df["site_numb"].fillna(
        site_df["site_numb"].values[0]
    )
    site_parent_df["parent_aa_loc"] = site_parent_df["parent_aa_loc"].fillna(
        site_df["parent_aa_loc"].values[0]
    )

    # now merge the two dataframes
    return pd.concat([site_parent_df, site_df]).reset_index(drop=True).copy()


def prep_aa_order(df: pd.DataFrame, add_na: bool = False) -> pd.DataFrame:
    """
    Prepare the data for a single sitessm summary plot.

    Args:
    - df: pd.DataFrame, input full dataframe

    Returns:
    - pd.DataFrame, output dataframe
    """

    # Define the order of x-axis categories
    x_order = list(AA_DICT.values())

    if add_na:
        x_order += ["#N.A.#"]

    # Convert `Mutations` to a categorical column with specified order
    df["mut_aa"] = pd.Categorical(df["mut_aa"], categories=x_order, ordered=True)

    # Sort by the `x_order`, filling missing values
    return (
        df.sort_values("mut_aa", key=lambda x: x.cat.codes)
        .reset_index(drop=True)
        .copy()
    )


def get_parent2sitedict(df: pd.DataFrame) -> dict:

    """
    Get a dictionary of parent to site mapping for single site mutants.

    Args:
    - df : pd.DataFrame

    Returns:
    - dict
        A dictionary containing the parent sequence and site number for each parent.
    """

    site_dict = deepcopy(
        df[["Parent_Name", "parent_aa_loc"]]
        .drop_duplicates()
        .dropna()
        .groupby("Parent_Name")["parent_aa_loc"]
        .apply(list)
        .to_dict()
    )

    # Sort the site list for each parent as an integer
    for parent, sites in site_dict.items():
        # Ensure each site is processed as a string and sorted by the integer part
        site_dict[parent] = sorted(sites, key=lambda site: int(str(site)[1:]))

    return site_dict


def get_x_label(x: str):
    
    """
    Function to return the x-axis label based on the input string.
    """

    if "mut_aa" in x.lower():
        clean_x = x.replace("mut_aa", "Amino acid substitutions")
    else:
        clean_x = x.replace("_", " ").capitalize()

    return clean_x


def get_y_label(y: str):

    """
    Function to return the y-axis label based on the input string.
    """
    clean_y = ""
    if "pdt" in y.lower():
        clean_y = "Product"
    elif "area" in y.lower():
        clean_y = "Yield"
    elif y == "fitness_ee2/(ee1+ee2)":
        clean_y = "ee2/(ee1+ee2)"
    elif y == "fitness_ee1/(ee1+ee2)":
        clean_y = "ee1/(ee1+ee2)"
    else:
        clean_y = y

    # normalize the y label
    if "norm" in y.lower():
        clean_y = f"Normalized {clean_y.lower()}"
    return clean_y


def plot_bar_point(
    df: pd.DataFrame,
    x: str,
    y: str,
    x_label: str = None,
    y_label: str = None,
    title: str = None,
    if_max: bool = False,
) -> hv.Layout:

    # Create Bars plot
    bars = hv.Bars(
        df[[y, x]].sort_values(x).groupby(x).mean(),
        kdims=x,
        vdims=y,
    )

    # Display the plot
    bars.opts(
        title=title,
        xlabel=x_label or get_x_label(x),
        ylabel=y_label or get_y_label(y),
        color=y,
        cmap="coolwarm",
        width=600,
        height=400,
        xrotation=45,
    )

    # Create Scatter chart
    points = hv.Scatter(df, x, [y, "Plate", "Well"]).opts(
        color=y, cmap="gray", size=8, alpha=0.5, tools=["hover"]
    )

    # create another scatter plot to highlight the max value
    if if_max:
        max_points = hv.Scatter(
            df.loc[df.groupby(x)[y].idxmax()],
            x,
            [y, "Plate", "Well"],
        ).opts(color="orange", size=10, alpha=1, tools=["hover"])
        return bars * points * max_points

    else:
        return bars * points


def get_parent_plot(df: pd.DataFrame, y: str = "pdt_norm") -> hv.Bars:

    """
    Function to plot the max value by parent.

    Args:
    - df : pd.DataFrame
        A pandas DataFrame containing the data for all parents.
        The DataFrame should have the Parent_Name columns
    - y : str
        The column name for which the max value is to be calculated.

    Returns:
    - hv.Bars
        A holoviews Bars object containing the plot.
    """

    parent_summary = df.groupby("Parent_Name")[y].max().reset_index()
    return hv.Bars(parent_summary, kdims="Parent_Name", vdims=y).opts(
        title="Max Value by Parent", width=600, height=400
    )


def agg_parent_plot(df: pd.DataFrame, ys: list = ["pdt_norm"]) -> pn.Row:

    """
    Function to plot the max value by parent for different y metrics.

    Args:
    - df : pd.DataFrame
        A pandas DataFrame containing the data for all parents.
        The DataFrame should have the Parent_Name columns
    - ys : list
        The list of column name for which the max value is to be calculated.

    Returns:
    - hv.Bars
    """

    # find single site mutations
    # avg_parnet_plots = [get_parent_plot(y=y) for y in ys if y in df.columns]
    avg_parnet_plots = [
        plot_bar_point(
            df,
            x="Parent_Name",
            y=y,
            title=f"{get_y_label(y)} across parents",
            if_max=True,
        )
        for y in ys
        if y in df.columns
    ]

    if len(avg_parnet_plots) == 0:
        return None
    # elif len(avg_ssm_plots) == 1:
    #     return avg_ssm_plots[0]
    else:
        return pn.Row(*avg_parnet_plots)


def plot_single_ssm_avg(
    single_ssm_df: pd.DataFrame,
    parent_name: str,
    y: str = "pdt_norm",
    width: int = 600,
):
    """
    Function to plot single site mutations with average values.

    Parameters:
    - df: DataFrame containing mutation data.
    """

    sliced_df = prep_aa_order(
        single_ssm_df[single_ssm_df["Parent_Name"] == parent_name].copy()
    )

    height = max(20 * sliced_df["site_numb"].nunique() + 60, 160)

    return hv.HeatMap(
        data=sliced_df[["parent_aa_loc", "mut_aa", y]]
        .dropna()
        .groupby(by=["parent_aa_loc", "mut_aa"])
        .mean()
        .sort_values(
            ["parent_aa_loc", "mut_aa"],
            key=lambda col: col.str.extract(r"(\d+)$").fillna(0).astype(int).iloc[:, 0]
            if col.name == "parent_aa_loc"
            else col
        )
        .reset_index(),
        kdims=["mut_aa", "parent_aa_loc"],
        vdims=[y],
    ).opts(
        height=height,
        width=width,
        cmap="coolwarm",
        colorbar=True,
        colorbar_opts=dict(title=get_y_label(y), width=8),
        xrotation=45,
        title=f"Average single site substitution for {parent_name}",
        xlabel="Amino acid substitutions",
        ylabel="Position",
        invert_yaxis=True,
        tools=["hover"],
    )


def agg_single_ssm_exp_avg(
    single_ssm_df: pd.DataFrame,
    parent_name: str,
    ys: list = ["pdt_norm"],
):

    # find single site mutations
    avg_ssm_plots = [
        plot_single_ssm_avg(single_ssm_df=single_ssm_df, parent_name=parent_name, y=y)
        for y in ys
        if y in single_ssm_df.columns
    ]

    if len(avg_ssm_plots) == 0:
        return None
    # elif len(avg_ssm_plots) == 1:
    #     return avg_ssm_plots[0]
    else:
        return pn.Row(*avg_ssm_plots)


