# =============================================================================
# Statistical Utility Functions
#
# Description:
# Collection of statistical utility functions for data analysis, reporting,
# and visualization.
# =============================================================================


def summary_stat_table(
    data,
    variables,
    groups=None,
    perform_testing=False,
    remove_totals=False,
    rename=None,
):
    """
    Create a formatted cohort characteristics table

    For categorical variables, counts and percentages are reported for each
    category. For numeric variables, the mean and standard deviation are
    reported. The resulting table includes non-missing sample sizes for each
    variable and an overall cohort size row. Optionally, summary statistics
    can be stratified by a grouping variable if requested, with or without
    statistical testing for differences between groups.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset.
    variables : list[str]
        Variables to summarize.
    groups : str or None, default: None
        Name of a grouping variable. Results will be stratified by the
        categories of this variable.
    perform_testing : bool, default: False
        Whether to perform statistical testing and calculate effect size for
        differences in variables between groups. Only applies if a variable
        name is give to `groups`. The chi-squared and Kruskal-Wallis rank-sum
        tests are used for testing and categorical and numerical variables,
        respectively. Cramer's V is calculated for categorical variables, while
        Epsilon-Squared is calculated from the Kruskal-Wallis H statistic.
    remove_totals : bool, default: False
        Whether to remove the total N and statistic columns in the final
        output for situations where it does not make sense to have it.
    rename : dict or None, default: None
        Dictionary for renaming variables in table. Should be of the format
        {"originalName": "newName"}.

    Returns
    -------
    pandas.DataFrame
        Formatted summary table containing variable names, category/group
        labels, sample sizes, and summary statistics.
    """

    import pandas as pd
    import numpy as np
    from scipy.stats import chi2_contingency, kruskal
    from scipy.stats.contingency import association

    df = data.copy()

    # Make sure grouping variable is a category if given
    if groups is not None and ~isinstance(
        df[groups].dtype, pd.CategoricalDtype
    ):
        df[groups] = pd.Categorical(df[groups])

    # Create empty dataframe to store results
    tbl = pd.DataFrame()

    # Get summary statistics for each column in `variables`
    for col in variables:
        # Format variable name
        if rename is not None:
            name_formatted = rename[col]
        else:
            name_formatted = col.title().replace("_", " ")

        # Calculating non-missing observations
        total_non_missing = sum(data[col].notna())

        # If data type is category or string, get N (%)
        if isinstance(
            df[col].dtype, pd.CategoricalDtype
        ) or pd.api.types.is_string_dtype(df[col]):
            # Make sure variable is a category
            if ~isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = pd.Categorical(df[col])

            # Get number of blanks needed for total column
            blanks = [" "] * (len(df[col].cat.categories) - 1)

            # Calculate counts
            counts = df[col].value_counts(sort=False)

            # Get data and summary stats for variable
            res = pd.DataFrame(
                {
                    "Variable": [f"{name_formatted}, N (%)"] + blanks,
                    "Groups": list(df[col].cat.categories),
                    "Total N": [f"{total_non_missing:,}"] + blanks,
                    "Stat": counts.apply(
                        lambda x, n=total_non_missing: f"{x:,} ({x / n:.1%})"
                    ),
                },
            )

            # Calculate summary statistics for groups if given
            if groups is not None:
                for cat in df[groups].cat.categories:
                    # Subset data
                    df_sub = df[df[groups] == cat]

                    # Calculating non-missing observations
                    non_missing = sum(df_sub[col].notna())

                    # Calculate counts
                    counts = df_sub[col].value_counts(sort=False)

                    # Get data and summary stats and add to results
                    cat_formatted = cat.title().replace("_", " ")
                    res_sub = pd.DataFrame(
                        {
                            f"{cat_formatted} N": [f"{non_missing:,}"] + blanks,
                            f"{cat_formatted} Stat": counts.apply(
                                lambda x, n=non_missing: f"{x:,} ({x / n:.1%})"
                            ),
                        },
                    )
                    res = pd.concat([res, res_sub], axis=1)

                # Perform statistical testing if requested
                if perform_testing:
                    # Create contingency table
                    input_tbl = pd.crosstab(df[groups], df[col])

                    # Perform testing
                    _, pval, _, _ = chi2_contingency(input_tbl)
                    pval = f"{pval:.1E}" if pval < 0.01 else f"{pval:.2f}"

                    # Calculate effect size (Cramer's V)
                    effect_size = association(input_tbl, method="cramer")

                    # Calculate effective degrees of freedom
                    rows, cols = input_tbl.shape
                    df_star = max(min(rows - 1, cols - 1), 1)

                    # Scale the base thresholds (0.1, 0.3, 0.5) based on
                    # effective degrees of freedom
                    scale_factor = np.sqrt(df_star)
                    thresh_s  = 0.1 / scale_factor
                    thresh_m = 0.3 / scale_factor
                    thresh_l  = 0.5 / scale_factor

                    # Format effect size with effect magnitude indicator
                    ## Negligible
                    if effect_size < thresh_s:
                        effect_size = f"{effect_size:.2f}"
                    ## Small effect
                    elif effect_size < thresh_m:
                        effect_size = f"{effect_size:.2f} +"
                    ## Moderate effect
                    elif effect_size < thresh_l:
                        effect_size = f"{effect_size:.2f} ++"
                    ## Large effect
                    else:
                        effect_size = f"{effect_size:.2f} +++"

                    # Add to results
                    res_test = pd.DataFrame(
                        {
                            "P-value": blanks + [pval],
                            "Effect Size": blanks + [effect_size],
                        },
                        index=list(res.index),
                    )
                    res = pd.concat([res, res_test], axis=1)

        # If data type is integer or float, get mean and standard deviation
        if pd.api.types.is_numeric_dtype(df[col]):
            # Calculate mean and standard deviation
            mean = round(np.mean(df[col]), 1)
            sd = round(np.std(df[col]), 1)

            # Get data and summary stats for variable
            res = pd.DataFrame(
                {
                    "Variable": [f"{name_formatted}, Mean±SD"],
                    "Groups": "-",
                    "Total N": f"{total_non_missing:,}",
                    "Stat": f"{mean:,}±{sd:,}",
                },
                index=[0],
            )

            # Calculate summary statistics for groups if given
            if groups is not None:
                test_input = []
                for cat in df[groups].cat.categories:
                    # Subset data
                    df_sub = df[df[groups] == cat]

                    # Calculating non-missing observations
                    non_missing = sum(df_sub[col].notna())

                    # Calculate mean and standard deviation
                    mean = round(np.mean(df_sub[col]), 1)
                    sd = round(np.std(df_sub[col]), 1)

                    # Get data and summary stats and add to results
                    cat_formatted = cat.title().replace("_", " ")
                    res_sub = pd.DataFrame(
                        {
                            f"{cat_formatted} N": f"{non_missing:,}",
                            f"{cat_formatted} Stat": f"{mean:,}±{sd:,}",
                        },
                        index=[0],
                    )
                    res = pd.concat([res, res_sub], axis=1)

                    # Save data for testing
                    test_input.append(df_sub[col])

                # Perform statistical testing if requested
                if perform_testing:
                    # Perform testing
                    h_stat, pval = kruskal(*test_input)
                    pval = f"{pval:.1E}" if pval < 0.01 else f"{pval:.2f}"

                    # Calculate effect size (Epsilon-Squared)
                    total_n = sum(list(map(len, test_input)))
                    effect_size = h_stat / (total_n - 1)

                    # Format effect size with effect magnitude indicator
                    ## Negligible
                    if effect_size < 0.01:
                        effect_size = f"{effect_size:.3f}"
                    ## Small effect
                    elif effect_size < 0.06:
                        effect_size = f"{effect_size:.2f} +"
                    ## Moderate effect
                    elif effect_size < 0.14:
                        effect_size = f"{effect_size:.2f} ++"
                    ## Large effect
                    else:
                        effect_size = f"{effect_size:.2f} +++"

                    # Add to results
                    res_test = pd.DataFrame(
                        {
                            "P-value": pval,
                            "Effect Size": effect_size,
                        },
                        index=[0],
                    )
                    res = pd.concat([res, res_test], axis=1)

        # Add results for variable to full table
        tbl = pd.concat([tbl, res], ignore_index=True)

    # Add overall N row at top of results once done with variables
    res = pd.DataFrame(
        {
            "Variable": "Overall",
            "Groups": "-",
            "Total N": f"{data.shape[0]:,}",
            "Stat": "-",
        },
        index=[0],
    )

    if groups is not None:
        for cat in df[groups].cat.categories:
            df_sub = df[df[groups] == cat]
            cat_formatted = cat.title().replace("_", " ")
            res_sub = pd.DataFrame(
                {
                    f"{cat_formatted} N": f"{df_sub.shape[0]:,}",
                    f"{cat_formatted} Stat": "-",
                },
                index=[0],
            )
            res = pd.concat([res, res_sub], axis=1)

        if perform_testing:
            res_test = pd.DataFrame(
                {
                    "P-value": "-",
                    "Effect Size": "-",
                },
                index=[0],
            )
            res = pd.concat([res, res_test], axis=1)

    tbl = pd.concat([res, tbl], ignore_index=True)

    # Remove total N and stats columns if requested
    if remove_totals:
        tbl = tbl.drop(columns=["Total N", "Stat"])

    return tbl


def kaplan_meier(
    data,
    event,
    time,
    positive_event,
    negative_event,
    groups=None,
    at_risk_table=False,
    logrank=True,
    x_label="Time",
    y_label="Survival Probability",
    plot_title=None,
    color_list=None,
    figsize=(6.5, 4.5),
):
    """
    Perform Kaplan-Meier-based survival analysis.

    Produces either a single Kaplan-Meier curve or multiple curves stratified by
    the variable provided to `groups`. The event variable is converted to a
    binary indicator where `positive_event` represents the observed event and
    `negative_event` represents censoring. Will perform logrank or multivariate
    logrank testing if requested.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing the event, time, and optional grouping
        variables.
    event : str
        Name of the column containing event status.
    time : str
        Name of the column containing follow-up time or time to event.
    positive_event : str or int
        Value in `event` indicating that the event was observed, such as
        "Dead" or 1.
    negative_event : str or int
        Value in `event` indicating that the event was not observed, such as
        "Alive" or 0.
    groups : str or None, default=None
        Name of the column used to stratify Kaplan-Meier curves. If None, a
        single overall survival curve is plotted.
    at_risk_table : bool, default=False
        Whether to add the at risk counts table to the plot.
    logrank : bool, default=False
        Whether to perform logrank or multivariate logrank test and add
        results to plot.
    x_label : str, default="Time"
        Label for the x-axis.
    y_label : str, default="Overall Survival"
        Label for the y-axis.
    plot_title : str or None, default=None
        Title for the plot.
    color_list : list[str], default=None
        List of matplotlib recognized colors. If left None, `Dark2` palette
        will be used.
    figsize : tuple[float, float], default=(6.5, 4.5)
        Width and height of the matplotlib figure in inches.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Matplotlib Figure and Axes objects containing the Kaplan-Meier plot.
    """

    import pandas as pd
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts
    from lifelines.statistics import multivariate_logrank_test

    pd.set_option("future.no_silent_downcasting", True)

    # Extract plotting data
    if groups is not None:
        df_surv = data[[event, time, groups]].dropna()
    else:
        df_surv = data[[event, time]].dropna()

    # Convert event to binary variable if not already
    if not (positive_event == 1 and negative_event == 0):
        df_surv[event] = (
            df_surv[event]
            .astype("str")
            .replace({positive_event: "1", negative_event: "0"})
            .astype("int")
        )

    # Get color list if not specified
    if color_list is None:
        color_list = plt.cm.Dark2.colors

    # Start plot
    fig, ax = plt.subplots(figsize=figsize)

    # Fit Kaplan-Meier curves
    kmf_list = []
    if groups is not None:
        for group, color in zip(df_surv[groups].unique(), color_list):
            # Subset data for current group
            df_surv_sub = df_surv[df_surv[groups] == group]

            # Set the Kaplan-Meier fitter and fit to data
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=df_surv_sub[time],
                event_observed=df_surv_sub[event],
                label=group,
            )
            kmf_list.append(kmf)

            # Plot curve
            kmf.plot_survival_function(
                ax=ax,
                ci_show=True,
                ci_alpha=0.2,
                linewidth=2,
                color=color,
                alpha=0.7,
            )

            # Set axis limits
            ax.set_xlim(left=0)
            ax.set_ylim(top=1, bottom=0)

            # Anchor legend
            ax.legend(loc="lower left")

    else:
        # Set the Kaplan-Meier fitter and fit to data
        kmf = KaplanMeierFitter()
        kmf.fit(durations=df_surv[time], event_observed=df_surv[event])
        kmf_list.append(kmf)

        # Plot curve
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            ci_alpha=0.2,
            linewidth=2,
            color="black",
            alpha=0.7,
        )

        # Set axis limits
        ax.set_xlim(left=0)
        ax.set_ylim(top=1, bottom=0)

    # Add median survival time indicators to plot
    # NOTE: if median survival time not reached, then medians come back
    # as `inf` and nothing will be plotted
    for kmf in kmf_list:
        median = kmf.median_survival_time_
        ax.vlines(
            x=median,
            ymin=0,
            ymax=0.5,
            linestyle="--",
            color="black",
            alpha=0.5,
        )
        ax.hlines(
            y=0.5,
            xmin=0,
            xmax=median,
            linestyle="--",
            color="black",
            alpha=0.5,
        )

    # Add at risk counts below curves if requested
    if at_risk_table:
        add_at_risk_counts(*kmf_list, ax=ax, rows_to_show=["At risk"])

    # Perform log-rank test and add to plot if requested
    if logrank and groups is not None:
        # Perform testing
        lrtest = multivariate_logrank_test(
            event_durations=df_surv[time],
            groups=df_surv[groups],
            event_observed=df_surv[event],
        )

        # Make p-value text better looking
        if lrtest.p_value < 0.001:
            p_label = "p < 0.001"
        else:
            p_label = f"p = {lrtest.p_value:.3f}"

        # Add results to plot
        plt.text(
            s=f"Log-rank: x2 = {lrtest.test_statistic:.1f}; {p_label}",
            x=0.5,
            y=0.9,
            fontsize=10,
            color="black",
            bbox={"facecolor": "white"},
            transform=ax.transAxes,
        )

    # Set plot titles
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if plot_title is not None:
        ax.set_title(plot_title)

    # Set plot theme (like theme_bw() in ggplot)
    ax.set_facecolor("white")
    ax.grid(True, color="#D0D0D0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")

    # Return figure and axes
    return fig, ax
