import streamlit as st
import pandas as pd
from threading import Thread
import plotly.express as px
from levseq.seqfit import normalise_calculate_stats, process_files

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

df, variant_df, plate_df = None, None, None

with c1:
    variant = st.file_uploader(
        "Variant file (output from LevSeq)",
        key="1",
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    )
with c2:
    fitness = st.file_uploader(
        "Fitness file",
        key="2",
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    )

with c1:
    if variant is not None:
        variant = pd.read_csv(variant)

with c2:
    if fitness is not None:
        plate_df = pd.read_csv(fitness, header=1)
        products = st.multiselect('Select products', list(set(plate_df['Compound Name'].values)))
        plate_name = str(fitness.name).split('.')[0]

padd3, c0, padd4 = st.columns([1, 6, 1])

def format_dataframes(variant, df):
    variant['id'] = [f'{p}_{w}' for p, w in variant[['Plate', 'Well']].values]
    df['id'] = [f'{p}_{w}' for p, w in df[['Plate', 'Well']].values]
    df.set_index('id', inplace=True)
    variant.set_index('id', inplace=True)

    df = df.join(variant, rsuffix='_processed_plate_df', how='outer')
    df['Type'] = [m if '*' not in str(v) else '#TRUNCATED#' for m, v in
                  df[['amino-acid_substitutions', 'aa_sequence']].values]

    df['Type'] = [v if v != '-' else '#DELETION#' for v in df['Type'].values]
    df['Type'] = [v if str(v)[0] == '#' else '#VARIANT#' for v in df['Type'].values]
    df['Type'] = [v if v != '#DELETION#' else 'Deletion' for v in df['Type'].values]
    df['Type'] = [v if v != '#VARIANT#' else 'Variant' for v in df['Type'].values]
    df['Type'] = [v if v != '#PARENT#' else 'Parent' for v in df['Type'].values]
    df['Type'] = [v if v != '#TRUNCATED#' else 'Truncated' for v in df['Type'].values]
    df['Type'] = [v if v != '#LOW#' else 'Low' for v in df['Type'].values]
    df['Type'] = [v if v != '#N.A.#' else 'Empty' for v in df['Type'].values]

    return df


def make_alignment_plot(df):
    # Streamlit app
    st.title("Alignment Count Plot")
    st.write("This is a histogram showing alignment counts categorized by type.")

    df['size'] = [m + 1 if isinstance(m, int) and m > 0 else 1 for m in df['# Mutations']]
    # Define custom color mapping
    color_mapping = {
        "Empty": "lightgrey",
        "Low": "#A6A7AC",
        "Deletion": "#6E6E6E",
        "Truncated": "#FCE518",
        "Parent": "#97CA43",
        "Variant": "#3A578F"
    }

    # Create the Plotly histogram
    fig = px.histogram(
        df,
        x="Alignment Count",
        color="Type",
        color_discrete_map=color_mapping,
        barmode="stack",
        category_orders={"Type": ["Empty", "Low", "Deletion", "Truncated", "Parent", "Variant"]},
        title="Alignment Counts"
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

def make_scatter_plot(df):
    # If there are more than 1 in the fitness, plot both.
    # ---------------------------------------------------------
    if len(products) > 1:
        prod_1 = products[0]
        prod_2 = products[1]
    else:
        prod_1 = products[0]
        prod_2 = 'Alignment Count'

    st.title("Scatter plot of multiple features")
    st.write("Either your first two features otherwise the only feature and the alignment count.")

    # Define the color palette as a dictionary for Plotly
    color_mapping = {
        "Empty": "lightgrey",
        "Low": "#A6A7AC",
        "Deletion": "#6E6E6E",
        "Truncated": "#FCE518",
        "Parent": "#97CA43",
        "Variant": "#3A578F"
    }

    # Create the Plotly scatter plot
    fig = px.scatter(
        df,
        x=prod_1,
        y=prod_2,
        color='Type',
        color_discrete_map=color_mapping,
        size='size',
        category_orders={"Type": ['Empty', 'Low', 'Deletion', 'Truncated', 'Parent', 'Variant']},
        title=f"{prod_1} vs {prod_2}"
    )

    # Update axis ranges and ticks
    fig.update_xaxes(title=f"{prod_1}")
    fig.update_yaxes(title=f"{prod_2}")

    # Adjust legend position
    fig.update_layout(
        legend=dict(
            title="Type",
            x=1.05,
            y=1,
            xanchor="left",
            yanchor="top"
        )
    )
    st.plotly_chart(fig)


def seqfit_runner():
    st.info(f'Running LevSeq using your files! Results will appear below shortly.')

    if plate_df is None or variant is None:
        st.error("Please upload both Variant and Fitness files.")
        return

    # Process variants
    df = process_files(variant, plate_df, plate_name, products)

    df['# Mutations'] = [len(str(m).split('_')) if m not in ['#N.A.#', '#PARENT#', '-', '#LOW#'] else 0 for m in
                         df['amino-acid_substitutions'].values]

    # ------------------------ Display paired seq function data as a table
    st.title("Joined sequence function data")
    st.dataframe(df)  # Interactive table with scrollbars

    # ------------------------ Stats
    value_columns = products
    stats_df = pd.DataFrame()
    for value in value_columns:
        stats_df = pd.concat([stats_df, normalise_calculate_stats(df, [value], normalise='standard',
                                             stats_method='mannwhitneyu', parent_label='#PARENT#')])

    stats_df = stats_df.sort_values(by='amount greater than parent mean', ascending=False)
    # ------------------------ Display stats data as a table
    st.title("Statistics on the function data")
    st.dataframe(stats_df)  # Interactive table with scrollbars

    # -------------------------- Make visualisations
    df = format_dataframes(df, variant)

    make_alignment_plot(df)
    make_scatter_plot(df)

    st.subheader("Done LevSeq!")

def run_run():
    thread = Thread(target=seqfit_runner, args=())
    add_script_run_ctx(thread)
    thread.start()
    thread.join()

with c0:
    if c1 is not None and c2 is not None:
        st.button('Run LevSeq <3', on_click=run_run)
    else:
        st.info(
            f"""
               Upload a variant file and a fitness file.
            """
        )
        st.stop()