# =============================================================================
# Statistical Utility Functions
#
# Description:
# Collection of statistical utility functions for data analysis and reporting.
#
# =============================================================================


def summary_stat_table(data, cols):
    """
    Create a formatted cohort characteristics table.

    For categorical variables, counts and percentages are reported for each
    category. For numeric variables, the mean and standard deviation are
    reported. The resulting table includes non-missing sample sizes for each
    variable and an overall cohort size row.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset.
    cols : list[str]
        Variables to summarize.

    Returns
    -------
    pandas.DataFrame
        Formatted summary table containing variable names, category/group
        labels, sample sizes, and summary statistics.
    """

    import pandas as pd
    import numpy as np

    # Create empty dataframe to store results
    tbl = pd.DataFrame()

    # Get summary statistics for each column in `cols`
    for col in cols:
        # Nicely format variable name
        name_formatted = col.title().replace("_", " ")

        # Calculating non-missing observations
        non_missing = sum(data[col].notna())

        # If data type is category or string, get N (%)
        if data[col].dtype in ["category", "str"]:
            # Convert variable to category if string
            if data[col].dtype == "str":
                data[col] = data[col].astype("category")

            # Get number of blanks needed for total column
            blanks = [" "] * (len(data[col].cat.categories) - 1)

            # Calculate counts
            counts = data[col].value_counts(sort=False)

            # Get data and summary stats for variable
            res = pd.DataFrame(
                {
                    "Variable": [f"{name_formatted}, N (%)"] + blanks,
                    "Groups": list(data[col].cat.categories),
                    "Total N": [f"{non_missing:,}"] + blanks,
                    "Stat": counts.apply(
                        lambda x, n=non_missing: f"{x:,} ({x / n:.1%})"
                    ),
                },
            )

        # If data type is integer or float, get mean and standard deviation
        if data[col].dtype in ["int", "float"]:
            # Calculate mean and standard deviation
            mean = round(np.mean(data[col]), 1)
            sd = round(np.std(data[col]), 1)

            # Get data and summary stats for variable
            res = pd.DataFrame(
                {
                    "Variable": [f"{name_formatted}, Mean±SD"],
                    "Groups": "-",
                    "Total N": f"{non_missing:,}",
                    "Stat": f"{mean:,}±{sd:,}",
                },
                index=[0],
            )

        # Add blank row under variable results to break up table nicely
        res = pd.concat(
            [
                res,
                pd.DataFrame(
                    {
                        "Variable": " ",
                        "Groups": " ",
                        "Total N": " ",
                        "Stat": " ",
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

        # Add results for variable to full table
        tbl = pd.concat([tbl, res], ignore_index=True)

    # Add final overall row once done with variables
    tbl = pd.concat(
        [
            tbl,
            pd.DataFrame(
                {
                    "Variable": "Overall",
                    "Groups": "-",
                    "Total N": f"{data.shape[0]:,}",
                    "Stat": "-",
                },
                index=[0],
            ),
        ],
        ignore_index=True,
    )

    return tbl
