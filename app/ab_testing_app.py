from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import src.statistics as ab_stats

DATA_DIR = PROJECT_ROOT / "data"



# ---------------------------------------------------------------------
# Streamlit setup
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="A/B Testing Toolkit",
    page_icon="📊",
    layout="wide",
)

st.title("A/B Testing Toolkit")

st.markdown(
    """
    <style>
    /* Radio Button Titles */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] [data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] p {
        font-size: 20px !important;
        font-weight: bold !important;
        font-family: inherit !important;
    }

    /* Metric Cards */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] div,
    [data-testid="stMetricLabel"] p {
        font-size: 1.9rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        font-size: 1.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
### Two-Sample Comparison: Confidence Intervals for Difference in Means

Compare two groups using confidence intervals and hypothesis testing.
You can select a local file, fetch a dataset from a URL, use one from the repository's `data/` directory,
or enter summary statistics manually.
"""
)


# ---------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------

def available_data_files():
    """List supported files in data/."""

    if not DATA_DIR.exists():
        return []

    supported_extensions = {
        ".csv",
        ".dat",
        ".txt",
        ".tsv",
    }

    return sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )


@st.cache_data
def read_data_file(path_or_buffer):
    """
    Read CSV or delimited text data.
    sep=None allows pandas to infer comma, tab or whitespace delimiters.
    Lines beginning with # are ignored.
    """
    return pd.read_csv(
        path_or_buffer,
        sep=None,
        engine="python",
        comment="#",
    )


def clean_numeric_column(
    series,
    replace_zero_with_nan=False,
):
    """Convert a column to numeric and remove missing observations."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    if replace_zero_with_nan:
        values = values.replace(
            0,
            np.nan,
        )

    return values.dropna()


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------


def make_histogram(sample_1, sample_2, sample_1_name, sample_2_name, number_of_bins):
    """Create a histogram similar to the notebook version."""
    combined = pd.concat([sample_1, sample_2])
    minimum, maximum = combined.min(), combined.max()

    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5

    bins = np.linspace(minimum, maximum, number_of_bins + 1)
    font_size = 10
    figure, axis = plt.subplots(figsize=(7, 4))
    plt.setp(axis.spines.values(), linewidth=2)

    axis.tick_params(direction="in", pad=7, length=6, width=1.5, which="major", right=True, top=True)

    axis.hist(
        sample_1, bins=bins, color="white", edgecolor="black", linewidth=2,
        label=f"{sample_1_name}: μ = {sample_1.mean():.2f}, σ = {sample_1.std(ddof=1):.2f}"
    )
    axis.hist(
        sample_2, bins=bins, color="red", edgecolor="red", linewidth=2, alpha=0.6,
        label=f"{sample_2_name}: μ = {sample_2.mean():.2f}, σ = {sample_2.std(ddof=1):.2f}"
    )

    font = 12
    axis.set_ylabel("Number", fontsize=font) 
    axis.set_xlabel("Value", fontsize=font)
    axis.tick_params(axis='both', labelsize=font) 
    axis.legend(fontsize=0.8*font, loc="best")
    figure.tight_layout()
    return figure


def make_errorbar_plot(mean1, mean2, margin1, margin2, sample_1_name, sample_2_name, confidence_percent):
    """Create the notebook-style horizontal confidence-interval plot."""
    font_size = 13
    figure, axis = plt.subplots(figsize=(8.5, 3.0))
    plt.setp(axis.spines.values(), linewidth=2)

    axis.tick_params(direction="in", pad=7, length=6, width=1.5, which="major", right=True, top=True)
    y_positions = [0.30, 0.70]

    axis.errorbar(
        mean1, y_positions[0], xerr=margin1, fmt="o", color="red", ecolor="red",
        capsize=6, capthick=2, markersize=9, linewidth=2, label=rf"${mean1:.3f}\pm{margin1:.3f}$"
    )
    axis.errorbar(
        mean2, y_positions[1], xerr=margin2, fmt="o", color="green", ecolor="green",
        capsize=6, capthick=2, markersize=9, linewidth=2, label=rf"${mean2:.3f}\pm{margin2:.3f}$"
    )

    axis.set_ylim(0, 1)
    axis.set_yticks(y_positions)
    font = 12
    axis.set_yticklabels([sample_1_name, sample_2_name], fontsize=font))
    axis.set_xlabel(f"Mean values ({confidence_percent:.2f}% confidence)",fontsize=font))
    axis.tick_params(axis='both', labelsize=font) 
    axis.legend(fontsize=0.8*font, loc="best")
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------
# Input method selection
# ---------------------------------------------------------------------


input_method = st.radio(
    "Input from file source or summary statistics?",
    ["File Source", "Summary statistics"],
    horizontal=True,
)


sample_1_values = None
sample_2_values = None


# ---------------------------------------------------------------------
# File / Source Import Processing
# ---------------------------------------------------------------------

if input_method == "File Source":
    
    st.markdown("<p style='font-size: 20px; font-weight: bold; margin-bottom: 5px;'>Choose file destination source</p>", unsafe_allow_html=True)
    source_type = st.radio(
        "Select source type:",
        ["Local Repository Data", "Upload from Computer", "Import from URL"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    dataframe = None
    
    if source_type == "Local Repository Data":
        data_files = available_data_files()
        if not data_files:
            st.error(f"No supported files were found in `{DATA_DIR}`. Add CSV or DAT files.")
            st.stop()
        
        selected_filename = st.selectbox(
            "Select data file from project repository",
            [path.name for path in data_files],
        )
        try:
            dataframe = read_data_file(DATA_DIR / selected_filename)
        except Exception as error:
            st.error(f"Could not read `{selected_filename}`: {error}")
            st.stop()
            
    elif source_type == "Upload from Computer":
        uploaded_file = st.file_uploader("Choose a CSV or text data file", type=["csv", "txt", "dat", "tsv"])
        if uploaded_file is not None:
            try:
                dataframe = read_data_file(uploaded_file)
            except Exception as error:
                st.error(f"Could not read uploaded file: {error}")
                st.stop()
        else:
            st.info("Awaiting local computer file upload.")
            st.stop()
            
    elif source_type == "Import from URL":
        url_input = st.text_input("Dataset web URL", placeholder="https://example.com")
        if url_input:
            try:
                dataframe = read_data_file(url_input)
            except Exception as error:
                st.error(f"Could not read dataset from URL: {error}")
                st.stop()
        else:
            st.info("Please enter a valid remote URL address link.")
            st.stop()

    if dataframe is not None:
        with st.expander("Preview data", expanded=False):
            st.dataframe(dataframe, use_container_width=True)

        numeric_columns = [
            column for column in dataframe.columns
            if pd.to_numeric(dataframe[column], errors="coerce").notna().sum() >= 2
        ]

        if len(numeric_columns) < 2:
            st.error("The selected file source dataset must contain at least two numeric columns.")
            st.stop()

        # Place Sample 1 selection, Sample 2 selection, and Zero-replacement dropdown together in a single row
        config_columns = st.columns(3)

        with config_columns[0]:
            sample_1_name = st.selectbox("Sample 1 column", numeric_columns, index=0)

        with config_columns[1]:
            sample_2_name = st.selectbox("Sample 2 column", numeric_columns, index=1)

        with config_columns[2]:
            replace_zeroes_selection = st.selectbox(
                "Replace zeroes with missing values?",
                ["No", "Yes"],
                index=0,
            )
            replace_zeroes = (replace_zeroes_selection == "Yes")

        sample_1_values = clean_numeric_column(dataframe[sample_1_name], replace_zeroes)
        sample_2_values = clean_numeric_column(dataframe[sample_2_name], replace_zeroes)

        n1 = len(sample_1_values)
        mean1 = float(sample_1_values.mean())
        sd1 = float(sample_1_values.std(ddof=1))

        n2 = len(sample_2_values)
        mean2 = float(sample_2_values.mean())
        sd2 = float(sample_2_values.std(ddof=1))


# ---------------------------------------------------------------------
# Summary-statistics input
# ---------------------------------------------------------------------

else:
    sample_columns = st.columns(2)

    with sample_columns[0]:
        st.markdown("<p style='font-size: 20px; font-weight: bold; margin-bottom: 5px;'>Sample 1</p>", unsafe_allow_html=True)
        sample_1_name = st.text_input("Sample 1 name", value="Sample 1")
        n1 = st.number_input("Sample 1 size", min_value=2, value=1622, step=1)
        mean1 = st.number_input("Sample 1 mean", value=75.6, format="%.6f")
        sd1 = st.number_input("Sample 1 standard deviation", min_value=0.0, value=9.8, format="%.6f")

    with sample_columns[1]:
        st.markdown("<p style='font-size: 20px; font-weight: bold; margin-bottom: 5px;'>Sample 2</p>", unsafe_allow_html=True)
        sample_2_name = st.text_input("Sample 2 name", value="Sample 2")
        n2 = st.number_input("Sample 2 size", min_value=2, value=1910, step=1)
        mean2 = st.number_input("Sample 2 mean", value=72.6, format="%.6f")
        sd2 = st.number_input("Sample 2 standard deviation", min_value=0.0, value=9.7, format="%.6f")


# ---------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------

st.divider()

control_row_1 = st.columns([1.2, 0.7, 1.4, 1.0])
with control_row_1[0]:
    st.markdown("**Confidence level [%]**")
with control_row_1[1]:
    confidence_percent = st.selectbox(
        "Confidence level", [90.0, 95.0, 99.0, 99.9, 99.99, 99.999], index=1, label_visibility="collapsed"
    )
with control_row_1[2]:
    st.markdown("**One or two tailed test:**")
with control_row_1[3]:
    sidedness = st.selectbox(
        "One or two tailed test", ["One-tailed", "Two-tailed"], index=1, label_visibility="collapsed"
    )

control_row_2 = st.columns([2.7, 0.8, 2.1, 0.6])
with control_row_2[0]:
    st.markdown("**Sample sizes for t- to z-distribution threshold:**")
with control_row_2[1]:
    z_threshold = st.selectbox(
        "Sample size threshold", [10, 20, 30, 50, 100], index=2, label_visibility="collapsed"
    )
with control_row_2[2]:
    st.markdown("**Assume equal sample variances:**")
with control_row_2[3]:
    equal_variances_choice = st.selectbox(
        "Assume equal sample variances", ["y", "n"], index=0, label_visibility="collapsed"
    )

equal_variances = (equal_variances_choice == "y")


# ---------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------

try:
    result = ab_stats.compare_means(
        n1=int(n1), mean1=float(mean1), sd1=float(sd1),
        n2=int(n2), mean2=float(mean2), sd2=float(sd2),
        confidence_percent=float(confidence_percent),
        sidedness=sidedness, equal_variances=equal_variances, z_threshold=int(z_threshold),
    )
except Exception as error:
    st.error(str(error))
    st.stop()


# ---------------------------------------------------------------------
# Results text
# ---------------------------------------------------------------------

st.markdown(f"##### {sample_1_name}: n = {int(n1)}, mean = {mean1:.3f}, standard deviation = {sd1:.3f}")
st.markdown(f"##### {sample_2_name}: n = {int(n2)}, mean = {mean2:.3f}, standard deviation = {sd2:.3f}")

if result["distribution"] == "z":
    st.markdown(f"##### Both sample sizes > {z_threshold} so using z-statistics")
else:
    st.markdown(f"##### At least one sample size ≤ {z_threshold} so using t-statistics")
    if np.isfinite(result["variance_ratio"]):
        st.write(
            f"Variance ratio is {result['variance_ratio']:.2f}. "
            "The notebook uses 0.5 to 2 as a rule-of-thumb range for similar variances."
        )

critical_name = "z-value" if result["distribution"] == "z" else "t-value"
st.markdown(
    f"##### At {confidence_percent:.4f}% confidence the {critical_name} is {result['critical_value']:.3f}, "
    f"giving a difference in the means of {result['difference']:.2f} ± {result['difference_margin']:.2f} "
    f"({result['lower_bound']:.2f} to {result['upper_bound']:.2f})."
)

if result["significant"]:
    st.success(
        f"Range does not pass through zero, so we reject the null hypothesis "
        f"(the result is significant at {confidence_percent:.4f}% confidence)."
    )
else:
    st.info(
        f"Range passes through zero, so we cannot reject the null hypothesis "
        f"(the result is not significant at {confidence_percent:.4f}% confidence)."
    )

if result["difference"] - result["difference_margin"] > 0:
    st.write(f"**Summary: {sample_1_name} is significantly greater than {sample_2_name}.**")
elif result["difference"] + result["difference_margin"] < 0:
    st.write(f"**Summary: {sample_2_name} is significantly greater than {sample_1_name}.**")
else:
    st.write("**Summary: there is no statistically significant difference.**")


# ---------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Broad target for the metric label container, direct divs, and nested text */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] div,
    [data-testid="stMetricLabel"] p {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    
    /* Broad target to reduce the size of the numeric values */
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        font-size: 1.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

metric_columns = st.columns(3)
metric_columns[0].metric("Difference in means", f"{result['difference']:.3f}")
metric_columns[1].metric("p-value", f"{result['p_value']:.5g}")
metric_columns[2].metric("Test statistic", f"{result['test_statistic']:.3f}")



# ---------------------------------------------------------------------
# Histogram, when raw data are available
# ---------------------------------------------------------------------

if sample_1_values is not None and sample_2_values is not None:

    st.markdown("<p style='font-size: 20px; font-weight: bold; margin-bottom: 5px;'>Sample distributions</p>", unsafe_allow_html=True)
   
    number_of_bins = st.selectbox("Number of bins", [2, 5, 10, 20, 50, 100], index=2)

    histogram_figure = make_histogram(
        sample_1=sample_1_values, sample_2=sample_2_values,
        sample_1_name=sample_1_name, sample_2_name=sample_2_name,
        number_of_bins=number_of_bins,
    )
    st.pyplot(histogram_figure, use_container_width=False)
    plt.close(histogram_figure)


# ---------------------------------------------------------------------
# Error-bar plot
# ---------------------------------------------------------------------
st.markdown("<p style='font-size: 20px; font-weight: bold; margin-bottom: 5px;'><br>Mean confidence intervals</p><br>", unsafe_allow_html=True)


errorbar_figure = make_errorbar_plot(
    mean1=mean1, mean2=mean2,
    margin1=result["sample_1_margin"], margin2=result["sample_2_margin"],
    sample_1_name=sample_1_name, sample_2_name=sample_2_name,
    confidence_percent=confidence_percent,
)
st.pyplot(errorbar_figure, use_container_width=False)
plt.close(errorbar_figure)

# ---------------------------------------------------------------------
# Explanatory notes
# ---------------------------------------------------------------------

with st.expander("Assumptions and interpretation"):
    st.markdown(
        """
- The two groups are treated as independent samples.
- Observations should be independent.
- Samples should be approximately normally distributed, or large enough for the central limit theorem to apply
- Choosing unequal variances uses Welch's two-sample t-test.Statistical significance does not necessarily imply practical importance.
 - A paired before/after dataset requires a paired test and should not be analysed as two independent samples.
        """)
st.caption(f"Data directory: {DATA_DIR}")
