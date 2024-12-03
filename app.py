import streamlit as st
import re
import numpy as np
import pandas as pd
from threading import Thread
import plotly.express as px
import plotly.graph_objects as go
from levseq_vis.seqfit import (
    normalise_calculate_stats,
    process_plate_files,
    prep_aa_order,
    prep_single_ssm,
    get_parent2sitedict,
)

from streamlit.runtime.scriptrunner import add_script_run_ctx

# """
# The base of this app was developed from:
# https://share.streamlit.io/streamlit/example-app-csv-wrangler/
# """


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
st.subheader("Beta mode, for issues post [here](https://github.com/fhalab/LevSeq)")

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
        "Fitness files",
        key="2",
        accept_multiple_files=True,  # Allow multiple files
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    )

with c1:
    if seq_variant is not None:
        seq_variant = pd.read_csv(seq_variant)

with c2:

    fit_df_list = []  # List to store all the fitness files
    products_set = set()  # Set to hold unique product names
    plate_names = []  # List to hold the names of the plates

    if fitness_files:
        for fitness in fitness_files:
            try:
                plate_df = pd.read_csv(fitness, header=1).dropna(
                    subset=["Compound Name"]
                )
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
                "Select products",
                sorted(products_set),  # Sort for better UI experience
                help="Select one or more products to filter.",
            )
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
    st.plotly_chart(fig)


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

    SHPAE_LIST = ["circle", "diamond", "triangle-up", "square", "cross"]

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
    st.plotly_chart(fig)


def plot_bar_point(
    df, x, y, x_label=None, y_label=None, title=None, if_max=False, bar_color="blue"
):
    # Group data by `x` and calculate mean for bars
    bar_data = df.groupby(x).mean()[y].reset_index()

    # Create bar plot
    bar_trace = go.Bar(
        x=bar_data[x], y=bar_data[y], name="Average", marker_color=bar_color
    )

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
        xaxis_title=x_label or x,
        yaxis_title=y_label or y,
        # width=800,
        # height=500,
    )

    return fig


# def get_parent_plot(df, y="pdt_fold"):
#     parent_summary = df.groupby("Parent_Name")[y].max().reset_index()

#     fig = px.bar(
#         parent_summary,
#         x="Parent_Name",
#         y=y,
#         title="Max Value by Parent",
#         labels={"Parent_Name": "Parent", y: "Max Value"},
#         color="#97CA43",
#     )
#     fig.update_layout(width=800, height=500)
#     return fig


def agg_parent_plot(df, ys=["pdt_fold"]):
    # Create plots for each metric
    plots = [
        plot_bar_point(
            df,
            x="Parent_Name",
            y=y,
            bar_color="#97CA43",
            title=f"{y} across parents",
            if_max=True,
        )
        for y in ys
        if y in df.columns
    ]

    if not plots:
        return None

    # Display plots in Streamlit
    for plot in plots:
        st.plotly_chart(plot)


# def agg_mut_plot(site_df, site_info, parent, ys):
#     for y in ys:
#         if y in site_df.columns:
#             plot = plot_bar_point(
#                 site_df,
#                 x="mut_aa",
#                 y=y,
#                 title=f"{site_info} for {parent}",
#                 if_max=False
#             )
#             st.plotly_chart(plot)


def plot_single_ssm_avg(single_ssm_df, ys=["pdt_fold"]):
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
                    labels={"x": "Position", "y": "Mutation", "color": y},
                    title=f"Average Single Site Substitution for {parent_name}",
                    # height=600,
                    # width=width,
                )
                all_plots.append(fig)

    # Render all plots in Streamlit
    for fig in all_plots:
        st.plotly_chart(fig)


# def plot_single_ssm_avg(single_ssm_df, parent_name, y="pdt_fold", width=600):
#     # Filter data for the selected parent
#     sliced_df = prep_aa_order(
#         single_ssm_df[single_ssm_df["Parent_Name"] == parent_name].copy()
#     )

#     if sliced_df.empty:
#         st.warning("No data available for the selected parent.")
#         return None

#     # Prepare heatmap data
#     heatmap_data = sliced_df.pivot_table(
#         index="mut_aa",
#         columns="parent_aa_loc",
#         values=y,
#         aggfunc="mean",
#     ).T

#     # Create heatmap using Plotly
#     fig = px.imshow(
#         heatmap_data,
#         labels={"x": "Position", "y": "Mutation", "color": y},
#         title=f"Average Single Site Substitution for {parent_name}",
#         height=600,
#         width=width,
#         color_continuous_scale="RdBu",
#     )
#     fig.update_layout(xaxis=dict(tickangle=45), yaxis=dict(autorange="reversed"))
#     st.plotly_chart(fig)


# def agg_single_ssm_exp_avg(single_ssm_df, ys=["pdt_fold"]):
#     # Create a dropdown for parent selection
#     parents = single_ssm_df["Parent_Name"].unique()
#     parent_name = st.selectbox("Select Parent", options=parents)

#     # Generate plots for the selected parent
#     for y in ys:
#         if y in single_ssm_df.columns:
#             plot_single_ssm_avg(single_ssm_df=single_ssm_df, parent_name=parent_name, y=y)


def seqfit_runner():
    st.info(f"Running LevSeq using your files! Results will appear below shortly.")

    if fit_df is None or seq_variant is None:
        st.error("Please upload both Sequence and Fitness files.")
        return

    # Process variants
    df = process_plate_files(
        products=products, fit_df=fit_df, seq_df=seq_variant, plate_names=plate_names
    )

    fold_products = [f"{p}_fold" for p in products]
    parents_list = df["Parent_Name"].unique()

    # ------------------------ Display paired seq function data as a table
    st.title("Joined sequence function data (fold change wrt parent per plate)")
    st.dataframe(df)  # Interactive table with scrollbars

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
    # st.bokeh_chart(gen_seqfitvis(df, products))
    # Generate parent plot
    st.header("Parent Aggregation")
    agg_parent_plot(df, ys=fold_products)

    # Generate mutation plots
    st.header("Mutation Aggregation")
    single_ssm_df = prep_single_ssm(df)
    plot_single_ssm_avg(single_ssm_df, ys=fold_products)

    # # Generate single SSM plots
    st.header("Single SSM Heatmap")
    sites_dict = get_parent2sitedict(single_ssm_df)
    # agg_mut_plot(site_df, site_info="Site 1", parent="Parent 1", ys=["pdt_fold"])

    st.subheader("Done LevSeq!")


def run_run():
    thread = Thread(target=seqfit_runner, args=())
    add_script_run_ctx(thread)
    thread.start()
    thread.join()


with c0:
    if c1 is not None and c2 is not None:
        st.button("Run LevSeq <3", on_click=run_run)
    else:
        st.info(
            f"""
               Upload a variant file and a fitness file.
            """
        )
        st.stop()