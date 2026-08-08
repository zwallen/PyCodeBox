# =============================================================================
# Utility Functions for Plotting/Data Visualizations
#
# Description:
# Collection of utility functions for plotting/visualizing data.
# =============================================================================


def histogram(
    data,
    variable,
    groups=None,
    boxplot=False,
    x_label="Variable",
    y_label="Counts",
    plot_title=None,
    color_list=None,
    figsize=(6.5, 4.5),
):
    """
    Plot histogram distributions.

    Produces either a single histogram or multiple histograms stratified by
    the variable provided to `groups`.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset.
    variable : str
        Name of the variable to plot.
    groups : str or None, default=None
        Name of the column used to stratify distributions. If None, a single
        distribution is plotted for the full data.
    boxplot : bool, default=False
        Add horizontal boxplot(s) beneath the histogram.
    x_label : str, default="Variable"
        Label for the x-axis.
    y_label : str, default="Counts"
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
        Matplotlib Figure and Axes objects containing the histogram.
    """

    import matplotlib.pyplot as plt
    import math
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    # Create plotting areas
    if boxplot:
        # Set figure
        fig = plt.figure(figsize=figsize)

        # Specify grid specification for multiple plots
        gs = GridSpec(
            nrows=2,
            ncols=1,
            height_ratios=[4, 0.75],
            hspace=0.05,
        )

        # Specify the separate axes from the grid, setting boxplots to share
        # the same x-axis
        ax = fig.add_subplot(gs[0])
        ax_box = fig.add_subplot(gs[1], sharex=ax)

    else:
        # Set single figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        ax_box = None

    # Perform plotting
    if groups is not None:
        # Get list of colors based on number of groups
        color_list = plt.cm.Dark2.colors[: len(data[groups].unique())]

        # Plot histograms
        for group, color in zip(data[groups].unique(), color_list):
            # Subset data for group
            df_sub = data[data[groups] == group]

            # Plot histogram
            step_size = math.ceil(max(df_sub[variable]) * 0.03)
            ax.hist(
                df_sub[variable],
                bins=range(
                    min(df_sub[variable]),
                    max(df_sub[variable]) + step_size,
                    step_size,
                ),
                color=color,
                edgecolor="black",
                alpha=0.5,
                label=group,
            )

        # Set legend
        ax.legend(loc="best")

    else:
        # Plot histogram
        step_size = math.ceil(max(data[variable]) * 0.03)
        ax.hist(
            data[variable],
            bins=range(
                min(data[variable]),
                max(data[variable]) + step_size,
                step_size,
            ),
            color="grey",
            edgecolor="black",
        )

    # Add horizontal boxplot(s) if requested
    if boxplot:
        if groups is not None:
            # Get data for boxplot(s)
            box_data = [
                data.loc[data[groups] == group, variable].dropna()
                for group in data[groups].unique()
            ]

            # Plot boxplot(s)
            bp = ax_box.boxplot(
                box_data,
                vert=False,
                patch_artist=True,
                widths=0.6,
                label=data[groups].unique(),
            )
            ax_box.set_yticks([])

            # Set colors for boxes
            for patch, color in zip(bp["boxes"], color_list):
                patch.set_facecolor(color)
                patch.set_edgecolor("black")
                patch.set_alpha(0.5)

            # Set colors and line width for medians
            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(2)

        else:
            ax_box.boxplot(
                data[variable],
                vert=False,
                patch_artist=True,
                widths=0.6,
                boxprops={"facecolor": "lightgrey", "edgecolor": "black"},
                medianprops={"color": "black", "linewidth": 2},
                whiskerprops={"color": "black"},
                capprops={"color": "black"},
            )
            ax_box.set_yticks([])

        # Remove duplicate x-axis labels from histogram
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

        # Set x-axis label at boxplots
        ax_box.set_xlabel(x_label)

        # Set plot theme
        ax_box.set_facecolor("white")
        for spine in ax_box.spines.values():
            spine.set_color("black")

    # Expand breaks for histogram
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))

    # Format histogram y-axis to comma format
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    # Set plot titles
    if not boxplot:
        ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if plot_title is not None:
        ax.set_title(plot_title)

    # Set plot theme
    ax.set_facecolor("white")
    ax.grid(True, color="#D0D0D0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")

    # Return figure with double or single axes
    if boxplot:
        return fig, (ax, ax_box)
    return fig, ax


def kaplan_meier(
    data,
    event,
    time,
    positive_event,
    negative_event,
    groups=None,
    at_risk_table=False,
    logrank=False,
    x_label="Time",
    y_label="Overall Survival",
    plot_title=None,
    figsize=(6.5, 4.5),
):
    """
    Plot Kaplan-Meier survival curves.

    Produces either a single Kaplan-Meier curve or multiple curves stratified by
    the variable provided to `groups`. The event variable is converted to a
    binary indicator where `positive_event` represents the observed event and
    `negative_event` represents censoring.

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
            .replace({positive_event: "1", negative_event: "0"})
            .astype("int")
        )

    # Start plot
    fig, ax = plt.subplots(figsize=figsize)

    # Fit Kaplan-Meier curves
    kmf_list = []
    if groups is not None:
        # Get list of colors based on number of groups
        color_list = plt.cm.Dark2.colors[: len(df_surv[groups].unique())]

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

            # Set lower axis limits
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

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

        # Set lower axis limits
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    # Add median survival time indicators to plot
    # NOTE: if median survival time not reached, then medians come back
    # as `inf` and nothing will be plotted
    for kmf in kmf_list:
        median = kmf.median_survival_time_
        ax.vlines(
            x=median, ymin=0, ymax=0.5, linestyle="--", color="black", alpha=0.7
        )
        ax.hlines(
            y=0.5, xmin=0, xmax=median, linestyle="--", color="black", alpha=0.7
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
            s=f"Log-rank: x2 = {lrtest.test_statistic:.1f}, {p_label}",
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
