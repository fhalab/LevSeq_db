import streamlit as st
import re
import os
import pandas as pd
from threading import Thread
import plotly.express as px
import plotly.graph_objects as go
from rdkit import Chem
from levseq_vis_dev.seqfit import (
    normalise_calculate_stats,
    process_plate_files,
    append_xy,
    prep_aa_order,
    prep_single_ssm,
    get_parent2sitedict,
    get_single_ssm_site_df,
    get_x_label,
    get_y_label,
)

from streamlit.runtime.scriptrunner import add_script_run_ctx

# """
# The base of this app was developed from:
# https://share.streamlit.io/streamlit/example-app-csv-wrangler/
# """


# Set the config for the plotly charts
config = {
    "toImageButtonOptions": {
        "format": "svg",  # Set the download format to SVG
    }
}

SHPAE_LIST = ["circle", "diamond", "triangle-up", "square", "cross"]

# Function to validate SMILES string
def validate_smiles(smiles_string):
    """
    Validates if a string is a valid SMILES using RDKit.
    Returns True if valid, False otherwise.
    """
    if not smiles_string:
        return False
    
    mol = Chem.MolFromSmiles(smiles_string)
    return mol is not None

# if old version of the file, rename the columns
LevSeq_cols = {
    "plate": "Plate",
    "well": "Well",
    "nucleotide_mutation": "Variant",
    "alignment_count": "Alignment Count",
    "average_mutation_frequency": "Average mutation frequency",
    "p_value": "P value",
    "p_adj_value": "P adj. value",
    "amino_acid_substitutions": "amino-acid_substitutions",
    "alignment_probability": "Alignment Probability"
}


def _max_width_():
    max_width_str = f"max-width: 1800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


st.title("LevSeq sequence function pairing")
st.subheader(
    "Beta mode, for issues post [here](https://github.com/fhalab/LevSeq) or via [email](mailto:levseqdb@gmail.com)"
)
st.subheader(
    "Note! Your file name of your plate needs to match the plate name in the LevSeq file, see our [examples](https://github.com/fhalab/LevSeq) for details :) "
)
c1, c2 = st.columns([6, 6])

df, seq_variant, fitness_files = None, None, None

with c1:
    seq_variant = st.file_uploader(
        "Variant sequencing file (output from LevSeq)",
        key="1",
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    )
with c2:
    fitness_files = st.file_uploader(
        "Fitness files (support multiple files)",
        key="2",
        accept_multiple_files=True,  # Allow multiple files
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    )

with c1:
    if seq_variant is not None:
        seq_variant = pd.read_csv(seq_variant)
        # if old version of the file, rename the columns
        seq_variant = seq_variant.rename(columns=LevSeq_cols)

with c2:

    fit_df_list = []  # List to store all the fitness files
    products_set = set()  # Set to hold unique product names
    plate_names = []  # List to hold the names of the plates

    if fitness_files:
        for fitness in fitness_files:
            try:
                # Read the file into a DataFrame without assuming any header
                df = pd.read_csv(fitness)

                # check if multi index due to spaces in the header
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index()

                if "Compound Name" in df.columns:
                    # Use the column names as the header
                    pass

                # Check if the first entry contains "Compound name (signal)"
                elif "Compound name (signal)" in df.iloc[0].astype(str).values:
                    # Use the second row as the header
                    df.columns = df.iloc[1]  # Set the second row as the header
                    df = (
                        df[2:].reset_index(drop=True).copy()
                    )  # Drop the first two rows (metadata and headers)

                else:
                    # Use the first row as the header
                    df.columns = df.iloc[0]  # Set the first row as the header
                    df = (
                        df[1:].reset_index(drop=True).copy()
                    )  # Drop the first row (headers)

                plate_df = df.dropna(subset=["Compound Name"])
                plate_name = str(fitness.name).split(".")[0]

                # add a column for the plate name
                plate_df["Plate"] = plate_name

                fit_df_list.append(plate_df)  # Append DataFrame to the list
                plate_names.append(plate_name)  # Extract plate names
                products_set.update(
                    plate_df["Compound Name"].dropna().values
                )  # Add products to the set

            except Exception as e:
                st.warning(f"Could not read file {fitness.name}: {e}")

        # Check if there are valid DataFrames in the list
        if fit_df_list:
            fit_df = pd.concat(fit_df_list, ignore_index=True)
        else:
            st.warning("No valid fitness files were uploaded or parsed.")

        # Display multi-select dropdown for products
        if products_set:
            products = st.multiselect(
                "Select products (support multipe selection)",
                sorted(products_set),  # Sort for better UI experience
                help="Select one or more products to filter.",
            )
            
            # Dictionary to store SMILES for each selected product
            smiles_dict = {}
            
            # Create a SMILES input box for each selected product
            if products:
                st.subheader("Enter SMILES for selected products")
                
                # Use columns to make it compact but readable
                for product in products:
                    smiles_input = st.text_input(
                        f"SMILES for {product}",
                        key=f"smiles_{product}",
                        help="Enter a canonical SMILES string for this compound"
                    )
                    
                    # Validate SMILES
                    if smiles_input:
                        if validate_smiles(smiles_input):
                            st.success(f"Valid SMILES for {product}")
                            smiles_dict[product] = smiles_input
                        else:
                            st.error(f"Invalid SMILES for {product}. Please enter a valid SMILES string.")
                    else:
                        st.warning(f"Please enter a SMILES string for {product}")
                
                # Store the SMILES dictionary in session state
                st.session_state['smiles_dict'] = smiles_dict
        else:
            st.warning("No products found in the fitness files.")

padd3, c0, padd4 = st.columns([1, 6, 1])


def make_alignment_plot(df):
    # Streamlit app
    st.title("Alignment Count Plot")
    st.write("This is a histogram showing alignment counts categorized by type.")

    # Define custom color mapping
    color_mapping = {
        "Empty": "lightgrey",
        "Low": "#A6A7AC",
        "Deletion": "#6E6E6E",
        "Truncated": "#FCE518",
        "Parent": "#97CA43",
        "Variant": "#3A578F",
    }

    # Create the Plotly histogram
    fig = px.histogram(
        df,
        x="Alignment Count",
        color="Type",
        color_discrete_map=color_mapping,
        barmode="stack",
        category_orders={
            "Type": ["Empty", "Low", "Deletion", "Truncated", "Parent", "Variant"]
        },
        title="Alignment Counts",
    )

    # Customize layout
    fig.update_layout(
        xaxis_title="Alignment Count",
        yaxis_title="Count",
        title_x=0.5,
        legend_title="Type",
        template="simple_white",
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, config=config)


def make_scatter_plot(df, parents_list):
    # If there are more than 1 in the fitness, plot both.
    # ---------------------------------------------------------
    if len(products) > 1:
        prod_1 = products[0]
        prod_2 = products[1]
    else:
        prod_1 = products[0]
        prod_2 = "Alignment Count"

    st.title("Scatter plot of multiple features")
    st.write(
        "Either your first two features otherwise the only feature and the alignment count."
    )

    # Define the color palette as a dictionary for Plotly
    color_mapping = {
        "Empty": "lightgrey",
        "Low": "#A6A7AC",
        "Deletion": "#6E6E6E",
        "Truncated": "#FCE518",
        "Parent": "#97CA43",
        "Variant": "#3A578F",
    }

    # Define the order of the legend with parents first
    type_order = ["Empty", "Low", "Deletion", "Truncated", "Parent", "Variant"]

    shape_mapping = {
        p: s for p, s in zip(parents_list, SHPAE_LIST[: len(parents_list)])
    }

    df["size"] = [
        m + 1 if isinstance(m, int) and m > 0 else 1 for m in df["# Mutations"]
    ]

    # Create the Plotly scatter plot
    fig = px.scatter(
        df,
        x=prod_1,
        y=prod_2,
        color="Type",
        color_discrete_map=color_mapping,
        symbol="Parent_Name",
        symbol_map=shape_mapping,
        size="size",
        category_orders={
            "Parent_Name": parents_list,
            "Type": type_order,
        },
        hover_data=[
            "Type",
            "Parent_Name",
            "Plate",
            "Well",
            "amino-acid_substitutions",
            "# Mutations",
            prod_1,
            prod_2,
        ],
        title=f"{prod_1} vs {prod_2}",
    )

    # Update axis ranges and ticks
    fig.update_xaxes(title=f"{prod_1}")
    fig.update_yaxes(title=f"{prod_2}")

    # Adjust legend position
    fig.update_layout(
        legend=dict(title="Type", x=1.05, y=1, xanchor="left", yanchor="top")
    )
    st.plotly_chart(fig, config=config)


def plot_bar_point(
    df,
    x,
    y,
    x_label=None,
    y_label=None,
    title=None,
    if_max=False,
    bar_color=None,
    colorscale=None,
    highlight_label=None,  # New: Value to highlight
    highlight_color="white",  # New: Color for the highlighted bar
    showlegend=True,
):

    # Group data by `x` and calculate mean for bars
    bar_data = df.groupby(x).mean()[y].reset_index()

    if bar_color:
        bar_kwargs = dict(marker_color=bar_color)
    elif colorscale:
        # Define bar colors conditionally
        bar_colors = [
            highlight_color if label == highlight_label else value
            for label, value in zip(bar_data[x], bar_data[y])
        ]

        # Define bar line styles conditionally
        bar_lines = [
            {"color": "lightgray", "width": 2}
            if val == highlight_label
            else {"color": "white", "width": 0}
            for val in bar_data[x]
        ]

        bar_kwargs = dict(
            marker=dict(
                color=bar_colors,  # Color based on y-values
                colorscale=colorscale,  # Use the specified colorscale
                line=dict(
                    color=[line["color"] for line in bar_lines],
                    width=[line["width"] for line in bar_lines],
                ),  # Add outlines for highlighted bar
                showscale=True,  # Show color scale legend
                colorbar=dict(
                    title=dict(
                        text=get_y_label(y),
                        side="right",  # Move title to the left of the color bar
                        # standoff=15,  # Adjust the spacing
                    ),
                    thickness=15,  # Set the length of the color bar
                    outlinewidth=0,  # Remove the border of the color bar
                ),  # Add title to the color scale
            )
        )
    else:
        bar_kwargs = dict()

    # Create bar plot
    bar_trace = go.Bar(x=bar_data[x], y=bar_data[y], name="Average", **bar_kwargs)

    # Create scatter plot
    scatter_trace = go.Scatter(
        x=df[x],
        y=df[y],
        mode="markers",
        name="Points",
        marker=dict(size=8, color="gray", opacity=0.6),
        text=df[["Plate", "Well"]].astype(str).agg(", ".join, axis=1),  # Tooltip info
    )

    # Add max points (if requested)
    traces = [bar_trace, scatter_trace]
    if if_max:
        max_data = df.loc[df.groupby(x)[y].idxmax()]
        max_trace = go.Scatter(
            x=max_data[x],
            y=max_data[y],
            mode="markers",
            name="Max",
            marker=dict(size=10, color="orange"),
            text=max_data[["Plate", "Well"]].astype(str).agg(", ".join, axis=1),
        )
        traces.append(max_trace)

    # Combine all traces
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title or f"{x} vs {y}",
        xaxis=dict(
            title=x_label or get_x_label(x),
            showline=True,
            linewidth=1,
            linecolor="gray",
        ),
        yaxis=dict(
            title=y_label or get_y_label(y),
            showline=True,
            linewidth=1,
            linecolor="gray",  # Color of the axis line
        ),
        showlegend=showlegend,
    )

    return fig


def agg_parent_plot(df, ys):
    # Create plots for each metric
    plots = [
        plot_bar_point(
            df,
            x="Parent_Name",
            y=y,
            bar_color="#97CA43",
            title=f"{get_y_label(y)} across parents",
            if_max=True,
        )
        for y in ys
        if y in df.columns
    ]

    if not plots:
        return None

    # Display plots in Streamlit
    for plot in plots:
        st.plotly_chart(plot, config=config)


def agg_mut_plot(sites_dict, single_ssm_df, ys):
    """
    Create individual plots for each parent and their respective sites.

    Args:
    - sites_dict (dict): A dictionary where keys are parents and values are lists of sites.
    - single_ssm_df (pd.DataFrame): DataFrame containing the single site mutation data.
    - ys (list): List of columns to plot.
    """
    for parent, sites in sites_dict.items():
        st.subheader(f"Parent: {parent}")  # Section title for each parent

        for site in sites:
            # Preprocess the site-specific data
            site_df = prep_aa_order(
                get_single_ssm_site_df(single_ssm_df, parent=parent, site=site)
            )

            if site_df.empty:
                st.warning(f"No data available for Parent: {parent}, Site: {site}")
                continue  # Skip if there's no data for the site

            sub_aas = site_df["amino-acid_substitutions"].unique()
            # skip if there is only one AA other than the parent
            if len(sub_aas) == 2:

                sub_aa = sub_aas[0] if sub_aas[1] == "#PARENT#" else sub_aas[1]

                st.markdown(
                    f"Only {sub_aa} AA other than the parent for Parent: {parent}, Site: {site}"
                )
                name_col = [
                    "Parent_Name",
                    "amino-acid_substitutions",
                    "# Mutations",
                    "Type",
                ]

                # Aggregate with both mean and std for numerical columns
                aggregated_df = (
                    site_df[name_col + ys]
                    .groupby(name_col)
                    .agg({col: ["mean", "std"] for col in ys})
                    .reset_index()
                )

                # Flatten the multi-level columns for readability
                aggregated_df.columns = [
                    "_".join(filter(None, col)).strip()
                    if isinstance(col, tuple)
                    else col
                    for col in aggregated_df.columns
                ]

                # Display the aggregated DataFrame
                st.dataframe(aggregated_df)

                continue

            site_info = (
                site_df["parent_aa_loc"].unique()[0] if not site_df.empty else "Unknown"
            )

            # Generate plots for each `y` value
            for y in ys:
                if y in site_df.columns:
                    fig = plot_bar_point(
                        site_df,
                        x="mut_aa",
                        y=y,
                        title=f"Parent: {parent}, Site: {site_info}, Metric: {get_y_label(y)}",
                        if_max=False,
                        highlight_label=site_info[0],  # New: Value to highlight
                        highlight_color="white",  # New: Color for the highlighted bar
                        colorscale="RdBu_r",
                        showlegend=False,
                    )
                    st.plotly_chart(fig, config=config)


def plot_single_ssm_avg(single_ssm_df, ys):
    # Preprocess data for each parent
    parents = single_ssm_df["Parent_Name"].unique()

    # Check if the data is empty
    if single_ssm_df.empty:
        st.warning("No data available to plot.")
        return

    # Create a container for all the plots
    all_plots = []

    for parent_name in parents:
        for y in ys:
            if y in single_ssm_df.columns:
                sliced_df = prep_aa_order(
                    single_ssm_df[single_ssm_df["Parent_Name"] == parent_name]
                )
                # Skip if sliced_df is empty
                if sliced_df.empty:
                    continue

                # Prepare heatmap data
                heatmap_data = sliced_df.pivot_table(
                    index="mut_aa", columns="parent_aa_loc", values=y, aggfunc="mean"
                ).reindex(
                    columns=sorted(
                        sliced_df["parent_aa_loc"].dropna().unique(),
                        key=lambda x: int(re.search(r"(\d+)$", str(x)).group())
                        if re.search(r"(\d+)$", str(x))
                        else 0,
                    )
                )

                # Create Plotly heatmap
                fig = px.imshow(
                    heatmap_data.T,
                    color_continuous_scale="RdBu_r",
                    labels={
                        "x": "Amino acid substitutions",
                        "y": "Position",
                        "color": y,
                    },
                    title=f"Average Single Site Substitution for {parent_name}",
                )

                # Update colorbar settings
                fig.update_coloraxes(
                    colorbar_title=get_y_label(y),
                    colorbar=dict(
                        title_side="right",  # Move title to the left of the color bar
                    ),
                )

                # Remove horizontal gridlines
                fig.update_layout(
                    xaxis=dict(showgrid=False),  # Disable gridlines for x-axis
                    yaxis=dict(showgrid=False),  # Disable gridlines for y-axis
                )
                all_plots.append(fig)

    # Render all plots in Streamlit
    for fig in all_plots:
        st.plotly_chart(fig, config=config)


def plot_embxy(df: pd.DataFrame, product: str, parents_list: list):

    """
    Function to plot the x, y embedding coordinates colored by a specified metric using Plotly.

    Args:
    - df : pd.DataFrame
        A pandas DataFrame containing the data for the x, y embedding coordinates.
    - product : str
        The column name to be used for coloring the scatter plot.

    Returns:
    - fig : plotly.graph_objs.Figure
        A Plotly figure object containing the scatter plot.
    """

    shape_mapping = {
        p: s for p, s in zip(parents_list, SHPAE_LIST[: len(parents_list)])
    }

    # Create a scatter plot using Plotly Express
    fig = px.scatter(
        df,
        x="x_coordinate",
        y="y_coordinate",
        color=product,
        symbol="Parent_Name",
        symbol_map=shape_mapping,
        hover_data=["ID", "amino-acid_substitutions", "Parent_Name"],
        title=f"{product} PCA in the sequence embedding space",
        color_continuous_scale="RdBu_r",
        labels={
            "x_coordinate": "Embedding PCA x",
            "y_coordinate": "Embedding PCA y",
            product: product,
        },
    )
    
    # Customize layout
    fig.update_layout(
        width=800,
        height=600,
        coloraxis_colorbar={
            "title": product,
            "x": 1.1,  # Move the color bar to the right of the plot
        },
        legend={
            "x": 0,     # Position the legend on the left
            "y": 1,     # Align the legend to the top
            "xanchor": "left",
            "yanchor": "top",
        },
    )

    return fig


def agg_embxy(df: pd.DataFrame, products: list, parents_list: list):
    """
    Function to aggregate the x, y embedding coordinates colored by different metrics using Plotly.

    Args:
    - df : pd.DataFrame
        A pandas DataFrame containing the data for the x, y embedding coordinates.
    - products : list
        A list of column names for which the x, y embedding coordinates are to be colored.

    Returns:
    - figs : list
        A list of Plotly figure objects containing the scatter plots.
    """

    plots = [
        plot_embxy(df, product, parents_list)
        for product in products
        if product in df.columns
    ]

    if not plots:
        return None

    # Display plots in Streamlit
    for plot in plots:
        st.plotly_chart(plot, config=config)


def seqfit_runner(smiles_dict):
    status_text = st.empty()
    error_text = st.empty()
    status_text.info(f"Running LevSeq using your files! Results will appear below shortly.")

    if fit_df is None or seq_variant is None:
        error_text.error("Please upload both Sequence and Fitness files.")
        return
    
    try:
        import traceback
        status_text.info("Step 1: Processing plate files...")
        # Process variants
        try:
            df = process_plate_files(
                products=products, fit_df=fit_df, seq_df=seq_variant, plate_names=plate_names
            ).copy()
            status_text.success("✅ Plate files processed successfully")
        except Exception as e:
            error_trace = traceback.format_exc()
            error_text.error(f"Error in process_plate_files: {str(e)}\n\nTraceback:\n{error_trace}")
            return
        
        status_text.info("Step 2: Adding SMILES data...")
        # Add SMILES data to the dataframe
        for product, smiles in smiles_dict.items():
            df[f"{product}_SMILES"] = smiles
        status_text.success("✅ SMILES data added successfully")

        fold_products = [f"{p}_fold" for p in products]
        parents_list = df["Parent_Name"].unique()

        # We'll only show the standardized format below
        
        # Format data in desired.csv format for download with melting of multiple products
        def format_to_standard_csv(df, smiles_dict, seq_df):
            # First, prepare the base columns common for all products
            base_cols = {
                'plate': df['Plate'],
                'well': df['Well'],
                'amino_acid_substitutions': df['amino-acid_substitutions'],
                'alignment_count': df['Alignment Count'],
                'average_mutation_frequency': df['Average mutation frequency'],
                'nt_sequence': df['nt_sequence'],
                'aa_sequence': df['aa_sequence'],
                'parent_name': df['Parent_Name'],
                'number_mutations': df['# Mutations'],
                'type': df['Type']
            }
            
            # Extract barcode_plate, p_value, and p_adj_value from the sequencing CSV if available
            base_cols['barcode_plate'] = ""
            base_cols['p_value'] = ""
            base_cols['p_adj_value'] = ""
            
            # Check for barcode_plate in the sequencing CSV
            if 'barcode_plate' in seq_df.columns:
                # Create a mapping dictionary from plate+well to barcode_plate
                plate_well_to_barcode = {}
                for idx, row in seq_df.iterrows():
                    if 'Plate' in row and 'Well' in row and 'barcode_plate' in row:
                        key = (row['Plate'], row['Well'])
                        plate_well_to_barcode[key] = row['barcode_plate']
                
                # Map barcode_plates to base_cols
                barcode_plates = []
                for plate, well in zip(base_cols['plate'], base_cols['well']):
                    key = (plate, well)
                    barcode_plates.append(plate_well_to_barcode.get(key, ""))
                base_cols['barcode_plate'] = barcode_plates
            
            # Check if seq_df has the p-value columns
            if 'P value' in seq_df.columns:
                # Create a mapping dictionary from plate+well to p_value
                plate_well_to_pvalue = {}
                for idx, row in seq_df.iterrows():
                    if 'Plate' in row and 'Well' in row and 'P value' in row:
                        key = (row['Plate'], row['Well'])
                        plate_well_to_pvalue[key] = row['P value']
                
                # Map p_values to base_cols
                p_values = []
                for plate, well in zip(base_cols['plate'], base_cols['well']):
                    key = (plate, well)
                    p_values.append(plate_well_to_pvalue.get(key, ""))
                base_cols['p_value'] = p_values
            
            # Check if seq_df has the adjusted p-value columns
            if 'P adj. value' in seq_df.columns:
                # Create a mapping dictionary from plate+well to p_adj_value
                plate_well_to_padj = {}
                for idx, row in seq_df.iterrows():
                    if 'Plate' in row and 'Well' in row and 'P adj. value' in row:
                        key = (row['Plate'], row['Well'])
                        plate_well_to_padj[key] = row['P adj. value']
                
                # Map p_adj_values to base_cols
                p_adj_values = []
                for plate, well in zip(base_cols['plate'], base_cols['well']):
                    key = (plate, well)
                    p_adj_values.append(plate_well_to_padj.get(key, ""))
                base_cols['p_adj_value'] = p_adj_values
            
            # Define well sorting helper function that keeps products together
            def sort_by_well(df):
                # Extract unique wells in sorted order
                all_wells = []
                for well in df['well'].unique():
                    well_info = extract_well_info(well)
                    all_wells.append((well_info, well))
                
                # Sort wells by row and column (A1, A2, ..., B1, B2, ...)
                sorted_wells = [w for _, w in sorted(all_wells)]
                
                # Create a new sorted dataframe
                sorted_df = pd.DataFrame()
                
                # For each well, get all products in that well
                for well in sorted_wells:
                    well_rows = df[df['well'] == well]
                    sorted_df = pd.concat([sorted_df, well_rows])
                
                return sorted_df.reset_index(drop=True)
                
            # If there are no products, return a dataframe with just the base columns
            if not products:
                standard_df = pd.DataFrame(base_cols)
                standard_df['compound_smiles'] = ""
                standard_df['fitness'] = 0
                standard_df['fold_change'] = 0
                standard_df['id'] = range(len(standard_df))
                # Sort by well (A1, A2, etc.)
                standard_df = sort_by_well(standard_df)
                # Ensure columns are in the correct order
                ordered_columns = ['id', 'barcode_plate', 'plate', 'well', 'amino_acid_substitutions', 
                                  'alignment_count', 'average_mutation_frequency', 'p_value', 'p_adj_value', 
                                  'nt_sequence', 'aa_sequence', 'compound_smiles', 'fitness', 'parent_name', 
                                  'fold_change', 'number_mutations', 'type']
                standard_df = standard_df[ordered_columns]
                return standard_df
            
            # For each product, create a separate dataframe
            all_dfs = []
            
            # Define well sorting helper function
            def extract_well_info(well):
                # Extract row letter and column number
                if len(well) >= 2 and well[0].isalpha() and well[1:].isdigit():
                    row_letter = well[0].upper()
                    col_number = int(well[1:])
                    # Return a tuple for sorting (row letter, column number)
                    return (row_letter, col_number)
                return ('Z', 999)  # Default for invalid wells
            
            # Get sorted indexes to maintain well order (A1, A2, ..., B1, B2, ...)
            well_info = [(extract_well_info(well), i) for i, well in enumerate(base_cols['well'])]
            sorted_indices = [idx for _, idx in sorted(well_info)]
            
            # Reorder base_cols by sorted well order
            for key in base_cols:
                if isinstance(base_cols[key], (list, pd.Series)):
                    base_cols[key] = [base_cols[key][i] for i in sorted_indices]
            
            # For each product, create a dataframe with the sorted base columns
            for product in products:
                product_df = pd.DataFrame(base_cols)
                
                # Add product-specific columns
                if product in smiles_dict:
                    product_df['compound_smiles'] = smiles_dict[product]
                else:
                    product_df['compound_smiles'] = ""
                
                # Add fitness value from the product column
                if product in df.columns:
                    # Get values in the sorted order
                    fitness_vals = [df[product].iloc[i] for i in sorted_indices]
                    product_df['fitness'] = fitness_vals
                else:
                    product_df['fitness'] = 0
                
                # Add fold change from the product_fold column
                fold_product = f"{product}_fold"
                if fold_product in df.columns:
                    # Get values in the sorted order
                    fold_vals = [df[fold_product].iloc[i] for i in sorted_indices]
                    product_df['fold_change'] = fold_vals
                else:
                    product_df['fold_change'] = 0
                
                # Add to our list of dataframes
                all_dfs.append(product_df)
            
            # Concatenate all product dataframes
            standard_df = pd.concat(all_dfs, ignore_index=True)
            
            # Sort the dataframe by well
            standard_df = sort_by_well(standard_df)
            
            # Add id column
            standard_df['id'] = range(len(standard_df))
            
            # Ensure columns are in the correct order
            ordered_columns = ['id', 'barcode_plate', 'plate', 'well', 'amino_acid_substitutions', 
                              'alignment_count', 'average_mutation_frequency', 'p_value', 'p_adj_value', 
                              'nt_sequence', 'aa_sequence', 'compound_smiles', 'fitness', 'parent_name', 
                              'fold_change', 'number_mutations', 'type']
            standard_df = standard_df[ordered_columns]
            
            return standard_df
            
        # Create a formatted dataframe for download
        standard_df = format_to_standard_csv(df, smiles_dict, seq_variant)
        
        # Save to the streamlit-data directory
        os.makedirs("streamlit-data", exist_ok=True)
        standard_df.to_csv("streamlit-data/desired.csv", index=False)
        
        # Show the standard format data
        st.title("Standardized sequence-function data")
        st.dataframe(standard_df)
        
        # Add download button for standard format only
        from functionforDownloadButtons import download_button
        download_button(standard_df, 'levseq_results.csv', 'Download results (CSV)')

        # ------------------------ Stats
        value_columns = products
        stats_df = pd.DataFrame()
        for value in value_columns:
            stats_df = pd.concat(
                [
                    stats_df,
                    normalise_calculate_stats(
                        df,
                        [value],
                        normalise="standard",
                        stats_method="mannwhitneyu",
                        parent_label="#PARENT#",
                    ),
                ]
            )

        stats_df = stats_df.sort_values(
            by="amount greater than parent mean", ascending=False
        )
        # ------------------------ Display stats data as a table
        st.title("Statistics on the function data")
        st.dataframe(stats_df)  # Interactive table with scrollbars

        # -------------------------- Make visualisations

        make_alignment_plot(df)
        make_scatter_plot(df, parents_list)
        # Generate parent plot
        st.header("Parent Aggregation")
        agg_parent_plot(df, ys=fold_products)

        # Generate mutation plots
        st.header("Mutation Aggregation")
        single_ssm_df = prep_single_ssm(df)
        plot_single_ssm_avg(single_ssm_df, ys=fold_products)

        # Generate single SSM plots
        st.header("Single SSM by Parent and Site")
        agg_mut_plot(
            sites_dict=get_parent2sitedict(single_ssm_df),
            single_ssm_df=single_ssm_df,
            ys=fold_products,
        )

        # Generate embedding plot
        st.header("Sequence Embedding")
        try:
            st.subheader("esm2_t12_35M_UR50D PCA (will take a while)")
            # take out all the stop codon containing sequences
            df_xy = append_xy(df, products=fold_products)
            agg_embxy(df_xy, products=fold_products, parents_list=parents_list)
        except Exception as e:
            st.error(f"Error generating embedding plot: {str(e)}")
            st.info("Continuing with the rest of the analysis...")

        st.subheader("Done LevSeq!")
    except Exception as e:
        st.error(f"An error occurred during LevSeq processing: {str(e)}")


def run_run():
    # Use the global variables
    global products, fit_df, seq_variant, plate_names
    
    # Validate in the main thread before starting the background process
    if not products:
        st.error("Please select at least one product.")
        return
        
    if 'smiles_dict' not in st.session_state:
        st.error("Missing SMILES dictionary in session state.")
        return
        
    # Check that all products have valid SMILES
    smiles_dict = st.session_state['smiles_dict']
    missing_smiles = [p for p in products if p not in smiles_dict]
    
    if missing_smiles:
        st.error(f"Missing valid SMILES for products: {', '.join(missing_smiles)}")
        return
    
    # Don't use threading - it might be causing issues with Streamlit
    # Just run the function directly in the main thread
    st.info("Starting LevSeq processing...(running in main thread)")
    
    # Run directly without threading
    seqfit_runner(smiles_dict)


with c0:
    if c1 is not None and c2 is not None:
        # Check if all products have valid SMILES
        all_smiles_valid = True
        
        if 'products' in locals() and products:
            if 'smiles_dict' not in st.session_state:
                all_smiles_valid = False
            else:
                for product in products:
                    if product not in st.session_state.get('smiles_dict', {}):
                        all_smiles_valid = False
                        break
        
        if all_smiles_valid and products:
            st.button("Run LevSeq <3", on_click=run_run, key="run_button")
        else:
            st.warning("Please provide valid SMILES for all selected products before running.")
            st.button("Run LevSeq <3", disabled=True, key="disabled_button")
    else:
        st.info(
            f"""
               Upload a variant file and a fitness file.
            """
        )
        st.stop()