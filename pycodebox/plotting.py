#!/bin/env python
# -*- coding: utf-8 -*-


# ----------------------------------------------------------------#
# Plotting Utilities                                              #
# ----------------------------------------------------------------#
# These functions help in generating data visualizations.         #
# ----------------------------------------------------------------#


def stratified_barplot(data, var, strata, case_id, fill_color=None, xlab=None):
  """
  Creates a stratified bar plot showing the frequency distribution of a categorical
  variable across different strata, including an 'All cases' comparison group.

  Parameters:
  -----------
  data : pandas.DataFrame
    The DataFrame containing the variables of interest.
  var : str
    The name of the categorical variable in `data` to be plotted on the y-axis as
    frequencies.
  strata : str
    The name of the categorical variable in `data` to stratify by, displayed on the
    x-axis.
  case_id : str
    The name of the column in `data` used as the case identifier for counting
    observations.
  fill_color : dict or list, optional
    Color specification for the bars. Can be a dictionary mapping variable categories
    to colors, or a list of colors.
  xlab : str, optional
    The label for the x-axis. If not provided, it defaults to the name of the `strata`
    variable with underscores replaced by spaces.

  Returns:
  --------
  plotnine.ggplot
    A ggplot object containing the stratified bar plot.

  Features:
  --------
  - Bars showing frequency distribution of `var` across `strata` categories.
  - An 'All cases' group for comparison.
  - Frequency percentages displayed above bars.
  - Sample counts (N=x) displayed below x-axis.
  - Dodged bar positioning for multiple categories.

  Notes:
  ------
  - Missing values in `var` are automatically filtered out before plotting and an error
  arises if no data is left after filtering.
  """
  import pandas as pd
  import numpy as np
  import seaborn as sns
  import matplotlib.colors as mcolors
  from plotnine import (
    ggplot,
    aes,
    geom_bar,
    geom_text,
    scale_y_continuous,
    scale_fill_manual,
    coord_cartesian,
    labs,
    theme_classic,
    theme,
    element_text,
    position_dodge,
    guides,
    guide_legend,
  )

  # Check `var` is categorical
  if not (data[var].dtype.name == 'category'):
    print(
      'The variable supplied to `var` was not a pandas Categorical variable. \
      Converting this variable to pandas Categorical with no specific ordering of \
      categories.'
    )
    data[var] = pd.Categorical(data[var])

  # Check `strata` is categorical
  if not (data[strata].dtype.name == 'category'):
    print(
      'The variable supplied to `strata` was not a pandas Categorical variable. \
      Converting this variable to pandas Categorical with no specific ordering of \
      categories.'
    )
    data[strata] = pd.Categorical(data[strata])

  # Filter out missing values for `var`
  data = data[data[var].notna()]
  if data.empty:
    raise ValueError(
      'No data available for `var` and `strata` after filtering out missing \
      values of `var`.'
    )

  # Get counts for each combination of `var` and `strata`
  strat_data = data.groupby([strata, var], observed=True)[case_id].count().reset_index()

  # Ensure all combinations of categories are accounted for to avoid errors
  complete_combinations = pd.MultiIndex.from_product(
    [data[strata].cat.categories, data[var].cat.categories],
    names=[strata, var],
  )
  strat_data = (
    strat_data.set_index([strata, var]).reindex(complete_combinations).reset_index()
  )

  # Calculate frequencies
  total_counts = data.groupby(strata, observed=True)[case_id].count().reset_index()
  total_counts = dict(zip(total_counts[strata], total_counts[case_id]))

  strat_data['freq'] = strat_data.apply(
    lambda row: row[case_id] / total_counts[row[strata]]
    if pd.notna(row[case_id])
    else pd.NA,
    axis=1,
  )

  # Aggregate data for all cases
  full_data = data.groupby(var, observed=True)[case_id].count().reset_index()
  full_data[strata] = 'All cases'
  full_data['freq'] = full_data[case_id] / len(data)

  # Combine full and stratified data and remove any missing frequencies
  summ_stats = pd.concat([full_data, strat_data], ignore_index=True)
  summ_stats = summ_stats[summ_stats['freq'].notna()]

  # Make labels for counts and frequencies
  summ_stats['count_lab'] = summ_stats[case_id].apply(lambda x: f'N={x}')
  summ_stats['freq_lab'] = summ_stats['freq'].apply(lambda x: f'{round(x * 100, 1)}%')

  # Ensure `strata` categories has initial ordering
  summ_stats[strata] = pd.Categorical(
    summ_stats[strata],
    categories=['All cases'] + list(data[strata].cat.categories),
    ordered=True,
  )

  # Generate stratified bar plot
  if fill_color is None:
    fill_color = sns.color_palette('hls', n_colors=len(data[var].cat.categories))
    fill_color = [mcolors.to_hex(color) for color in fill_color]
  if xlab is None:
    xlab = strata[0].upper() + strata[1:].replace('_', ' ')
  g = (
    ggplot(summ_stats, aes(x=strata, y='freq', fill=var))
    + geom_bar(position='dodge', stat='identity', color='black')
    + geom_text(
      aes(y='freq + 0.1', label='freq_lab'),
      position=position_dodge(0.9),
      angle=90,
      size=12,
    )
    + geom_text(aes(y=-0.05, label='count_lab'), position=position_dodge(0.9), size=10)
    + scale_y_continuous(
      breaks=np.linspace(0, 1, 6), labels=lambda x: [f'{v:.0%}' for v in x]
    )
    + scale_fill_manual(
      name=var[0].upper() + var[1:].replace('_', ' '),
      values=fill_color,
    )
    + coord_cartesian(ylim=(-0.03, 1))
    + labs(x=xlab, y='Frequency (%)')
    + guides(
      fill=guide_legend(
        position='top',
        ncol=len(data[var].cat.categories),
      )
    )
    + theme_classic()
    + theme(
      text=element_text(size=12),
      axis_text_x=element_text(angle=20, hjust=1),
    )
  )

  return g


def stratified_violin_boxplot(data, var, strata, ylab=None, xlab=None):
  """
  Creates a violin box plot showing the distribution of a numerical variable across
  different strata, including an 'All cases' comparison group.

  Parameters:
  -----------
  data : pandas.DataFrame
    The DataFrame containing the variables of interest.
  var : str
    The name of the numerical variable in `data` to be plotted on the y-axis.
  strata : str
    The name of the categorical variable in `data` to stratify by, displayed on the
    x-axis.
  ylab : str, optional
    The title for the y-axis, describing the numerical variable. If not provided,
    it defaults to the name of the `var` variable with underscores replaced by spaces.
  xlab : str, optional
    The title for the x-axis. If not provided, it defaults to the name of the `strata`
    variable with underscores replaced by spaces.

  Returns:
  --------
  plotnine.ggplot
    A ggplot object containing the stratified violin box plot.

  Features:
  -------
  - Shows distribution of `var` across `strata` categories as data densities (violin
    plot) and the median + interquartile range (box plot).
  - The mean and 95% confidence interval for each stratum displayed as a red point
    range.
  - An 'All cases' group for comparison.
  - Median values displayed as text labels on the plot.

  Notes:
  ------
  - Missing values in `var` are automatically filtered out before plotting and an error
  arises if no data is left after filtering.
  """
  import pandas as pd
  import numpy as np
  from plotnine import (
    ggplot,
    aes,
    geom_violin,
    geom_boxplot,
    geom_text,
    geom_jitter,
    stat_summary,
    coord_cartesian,
    labs,
    theme_classic,
    theme,
    element_text,
  )

  # Check `var` is numerical
  if data[var].dtype.kind not in ['i', 'f']:
    raise ValueError('The variable supplied to `var` must be numeric.')

  # Check `strata` is categorical
  if not (data[strata].dtype.name == 'category'):
    print(
      'The variable supplied to `strata` was not a pandas Categorical variable. \
      Converting this variable to pandas Categorical with no specific ordering of \
      categories.'
    )
    data[strata] = pd.Categorical(data[strata])

  # Filter out missing values for `var`
  data = data[data[var].notna()]
  if data.empty:
    raise ValueError(
      'No data available for `var` and `strata` after filtering out missing \
      values of `var`.'
    )

  # Add data for an all cases group
  data = pd.concat([data, data])
  data[strata] = pd.Categorical(
    data[strata],
    categories=['All cases'] + data[strata].cat.categories.tolist(),
    ordered=True,
  )
  data.loc[
    data.reset_index().index.to_numpy() >= (data.shape[0] // 2),
    strata,
  ] = 'All cases'

  # Calculate median for each stratum of `strata`
  summ_stats = data.groupby(strata, observed=True)[var].median().reset_index()
  summ_stats['median_lab'] = summ_stats[var].apply(
    lambda x: f'Median={int(round(x, 0))}'
  )

  # Generate violin boxplot
  if ylab is None:
    ylab = var[0].upper() + var[1:].replace('_', ' ')
  if xlab is None:
    xlab = strata[0].upper() + strata[1:].replace('_', ' ')
  np.random.seed(1234)
  g = (
    ggplot(data, aes(y=var, x=strata))
    + geom_violin(scale='width', fill='lightgrey', color='black')
    + geom_boxplot(width=0.5, fill='white', color='black', outlier_size=0)
    + geom_jitter(width=0.05, color='black', size=3)
    + geom_text(
      summ_stats,
      aes(
        y=data[var].dropna().min()
        - (data.loc[data[var] > 0, var].dropna().min() * 1.5),
        label='median_lab',
      ),
      size=12,
    )
    + stat_summary(fun_data='mean_cl_normal', geom='pointrange', color='red', size=1)
    + coord_cartesian(
      ylim=(
        data[var].dropna().min() - (data.loc[data[var] > 0, var].dropna().min() * 1.5),
        data[var].dropna().max(),
      )
    )
    + labs(x=xlab, y=ylab)
    + theme_classic()
    + theme(
      text=element_text(size=12),
      axis_text_x=element_text(angle=20, hjust=1),
    )
  )

  return g


def stratified_coef_w_ci(
  data, var, strata, coef, lower, upper, pvalue=None, fill_color=None, xlab=None
):
  """
  Creates a coefficient plot showing the estimated coefficients and confidence
  intervals of a numerical variable across different strata.

  Parameters:
  -----------
  data : pandas.DataFrame
    The DataFrame in "long" format that contains columns denoting the names of the
    tested variables (`var`), groups tested (`strata`), coefficients (`coef`), and the
    lower (`lower`) and upper (`upper`) limits of the coefficient confidence interval.
    Optionally, it can also contain p-values (`pvalue`) for each coefficient.
  var : str
    The column in `data` denoting names of tested variables to be plotted on the y-axis.
    The plot will be faceted by this variable.
  strata : str
    The name of the categorical variable in `data` to stratify by, displayed as
    different color points and ranges.
  coef : float
    The column in `data` denoting the estimated coefficients for each stratum of
    `strata` and variable of `var`.
  lower : float
    The column in `data` denoting the lower limit of the confidence interval for each
    coefficient.
  upper : float
    The column in `data` denoting the upper limit of the confidence interval for each
    coefficient
  pvalue : float, optional
    The column in `data` denoting the p-values for each coefficient. If provided,
    p-values will be displayed as text labels on the plot. If not provided, no p-values
    will be displayed.
  fill_color : dict or list, optional
    Color specification for the points and ranges. Can be a dictionary mapping variable
    categories to colors, or a list of colors.
  xlab : str, optional
    The title for the x-axis describing the coefficients being plotted. If not provided,
    it defaults to the name of the `coef` variable with underscores replaced by spaces

  Returns:
  --------
  plotnine.ggplot
    A ggplot object containing the coefficient plot.

  Features:
  -------
  - Coefficients displayed as points with error bars representing confidence intervals.
  - An 'All cases' group for comparison.
  - P-values displayed as text labels on the plot if provided.
  """
  import pandas as pd
  import seaborn as sns
  import matplotlib.colors as mcolors
  from plotnine import (
    ggplot,
    aes,
    geom_vline,
    geom_errorbarh,
    geom_point,
    geom_text,
    position_dodge,
    scale_fill_manual,
    scale_color_manual,
    facet_grid,
    coord_cartesian,
    theme_bw,
    theme,
    element_text,
    element_blank,
    element_rect,
    labs,
  )

  # Check `var` is numerical
  if not (data[var].dtype.name == 'category'):
    print(
      'The variable supplied to `var` was not a pandas Categorical variable. \
      Converting this variable to pandas Categorical with no specific ordering of \
      categories.'
    )
    data[var] = pd.Categorical(data[var])

  # Check `strata` is categorical
  if not (data[strata].dtype.name == 'category'):
    print(
      'The variable supplied to `strata` was not a pandas Categorical variable. \
      Converting this variable to pandas Categorical with no specific ordering of \
      categories.'
    )
    data[strata] = pd.Categorical(data[strata])

  # Check `coef`, `lower`, and `upper` are numerical
  if data[coef].dtype.kind not in ['i', 'f']:
    raise ValueError('The variable supplied to `coef` must be numeric.')
  if data[lower].dtype.kind not in ['i', 'f']:
    raise ValueError('The variable supplied to `lower` must be numeric.')
  if data[upper].dtype.kind not in ['i', 'f']:
    raise ValueError('The variable supplied to `upper` must be numeric.')
  if pvalue and data[pvalue].dtype.kind not in ['i', 'f']:
    raise ValueError('The variable supplied to `pvalue` must be numeric.')

  # If `pvalue` provided, create p-value labels
  if pvalue:
    data['pvalue_labs'] = data[pvalue].apply(
      lambda x: f'P={x:.1e}' if x < 0.005 else f'P={x:.2f}' if x >= 0.005 else ''
    )

  # Generate coefficient plot
  if fill_color is None:
    fill_color = sns.color_palette('hls', n_colors=len(data[var].cat.categories))
    fill_color = [mcolors.to_hex(color) for color in fill_color]
  if xlab is None:
    xlab = coef[0].upper() + coef[1:].replace('_', ' ')
  g = (
    ggplot(data, aes(x=coef, y=strata, color=strata))
    + geom_vline(xintercept=(0 if data[coef].min() < 0 else 1), linetype='dashed')
    + geom_errorbarh(
      aes(xmin=lower, xmax=upper),
      position=position_dodge(0.3),
      height=0.01,
      size=1,
      show_legend=False,
    )
    + geom_point(position=position_dodge(0.3), size=3)
    + scale_fill_manual(
      name=strata[0].upper() + strata[1:].replace('_', ' '),
      values=fill_color,
    )
    + scale_color_manual(
      name=strata[0].upper() + strata[1:].replace('_', ' '),
      values=fill_color,
    )
    + facet_grid(f'{var} ~ .')
    + theme_bw()
    + theme(
      text=element_text(size=12),
      axis_text_y=element_blank(),
      axis_ticks_major_y=element_blank(),
      axis_title_y=element_blank(),
      axis_title_x=element_text(margin={'t': 20}),
      legend_text=element_text(size=10),
      legend_title=element_text(size=10),
      strip_text_y=element_text(angle=0),
      strip_background=element_rect(fill='lightgrey'),
    )
    + labs(x=xlab)
  )

  # Add p-value labels if provided
  if pvalue:
    g = (
      g
      + geom_text(
        aes(label='pvalue_labs'),
        x=data[upper].max() + (data[upper].max() * 0.05),
        ha='left',
        size=8,
        show_legend=False,
      )
      + coord_cartesian(
        xlim=(
          data[lower].min(),
          data[upper].max()
          + (
            (data[upper].max() * 0.4)
            if data[pvalue].min() < 0.005
            else (data[upper].max() * 0.3)
          ),
        )
      )
    )

  return g


def stratified_volcano(
  data,
  var,
  strata,
  coef,
  pvalue,
  pthresh=0.05,
  top_n=None,
  fill_color=None,
  xlab=None,
):
  """
  Creates a stratified volcano plot showing the association results between tested
  numerical variables across different strata. Useful for when there are too many
  variables to plot using the `stratified_coef_w_ci` function.

  Parameters:
  -----------
  data : pandas.DataFrame
    The DataFrame in "long" format that contains columns denoting the names of the
    tested variables (`var`), groups tested (`strata`), coefficients (`coef`), and the
    p-values for each coefficient (`pvalue`).
  var : str
    The column in `data` denoting names of tested variables to be plotted. Each variable
    will be an independent point on the plot.
  strata : str
    The name of the categorical variable in `data` to stratify by. The plot will be
    faceted by this variable.
  coef : float
    The column in `data` denoting the estimated coefficients for each stratum of
    `strata` and variable of `var`.
  pvalue : float
    The column in `data` denoting the p-values for each coefficient.
  pthresh : float, optional
    The p-value threshold to use for significance. Default is 0.05.
  top_n : float, optional
    The top number of enriched and depleted variables to show labels. Default is to
    show no labels.
  fill_color : dict or list, optional
    Color specification for the points and ranges. Can be a dictionary mapping variable
    categories to colors, or a list of colors.
  xlab : str, optional
    The title for the x-axis describing the coefficients being plotted. If not provided,
    it defaults to the name of the `coef` variable with underscores replaced by spaces

  Returns:
  --------
  matplotlib.figure.Figure
    A matplotlib figure object containing the volcano plot.

  Features:
  -------
  - Variables displayed as individual points plotted by their coefficient and p-value.
  """
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import matplotlib.colors as mcolors
  import seaborn as sns
  from adjustText import adjust_text

  # Check `var` is categorical
  if not (data[var].dtype.name == 'category'):
    print(
      'The variable supplied to `var` was not a pandas Categorical variable. \
      Converting this variable to pandas Categorical with no specific ordering of \
      categories.'
    )
    data[var] = pd.Categorical(data[var])

  # Check `strata` is categorical
  if not (data[strata].dtype.name == 'category'):
    print(
      'The variable supplied to `strata` was not a pandas Categorical variable. \
      Converting this variable to pandas Categorical with no specific ordering of \
      categories.'
    )
    data[strata] = pd.Categorical(data[strata])

  # Check `coef` and `pvalue` are numerical
  if data[coef].dtype.kind not in ['i', 'f']:
    raise ValueError('The variable supplied to `coef` must be numeric.')
  if data[pvalue].dtype.kind not in ['i', 'f']:
    raise ValueError('The variable supplied to `pvalue` must be numeric.')

  # Create enriched vs depleted column
  if min(data[coef]) < 0:
    data['direction'] = pd.Categorical(
      np.where(
        (data[coef] < 0) & (data[pvalue] < pthresh),
        'Depleted',
        np.where(
          (data[coef] > 0) & (data[pvalue] < pthresh),
          'Enriched',
          'Not significant',
        ),
      )
    )
  else:
    data['direction'] = pd.Categorical(
      np.where(
        (data[coef] < 1) & (data[pvalue] < pthresh),
        'Depleted',
        np.where(
          (data[coef] > 1) & (data[pvalue] < pthresh),
          'Enriched',
          'Not significant',
        ),
      )
    )

  # Create column for labels
  if top_n:
    data['var_labs'] = pd.NA
    data_sorted = data.sort_values(pvalue)
    for stratum in data[strata].cat.categories:
      data_subset = data_sorted[data_sorted[strata] == stratum]
      top_enriched = list(
        data_subset.loc[data_subset['direction'] == 'Enriched', var][0:top_n]
      )
      top_depleted = list(
        data_subset.loc[data_subset['direction'] == 'Depleted', var][0:top_n]
      )
      data['var_labs'] = np.where(
        data[var].isin(top_enriched + top_depleted) & (data[strata] == stratum),
        data[var],
        data['var_labs'],
      )

  # Create column for -log10 of the p-value
  data['log10p'] = -np.log10(data[pvalue])

  # Set up colors
  if fill_color is None:
    colors = sns.color_palette('hls', n_colors=len(data['direction'].cat.categories))
    fill_color = [mcolors.to_hex(color) for color in colors]

  # Create color mapping
  color_map = dict(zip(data['direction'].cat.categories, fill_color))

  # Set up x-axis label
  if xlab is None:
    xlab = coef[0].upper() + coef[1:].replace('_', ' ')

  ### Begin figure generation ###

  # Create figure and subplots
  fig, axes = plt.subplots(
    nrows=len(data[strata].cat.categories),
    ncols=1,
    figsize=(5, 4 * len(data[strata].cat.categories)),
    sharex=True,
    sharey=True,
  )
  if len(data[strata].cat.categories) == 1:
    axes = [axes]

  # Plot each stratum
  for i, stratum in enumerate(data[strata].cat.categories):
    ax = axes[i]
    strat_data = data[data[strata] == stratum]

    # Plot points for each direction category
    texts = []
    for direction in data['direction'].cat.categories:
      direction_data = strat_data[strat_data['direction'] == direction]
      if not direction_data.empty:
        ax.scatter(
          x=direction_data[coef],
          y=direction_data['log10p'],
          label=direction,
          c=color_map[direction],
          alpha=0.6,
          s=50,
        )

        # Add labels for top_n points if specified
        if top_n:
          labeled_data = direction_data[direction_data['var_labs'].notna()]
          for _, row in labeled_data.iterrows():
            text = ax.annotate(
              text=row['var_labs'],
              xy=(row[coef], row['log10p']),
              fontsize=8,
              ha='center',
              va='center',
              bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                alpha=0.6,
                edgecolor='black',
              ),
            )
            texts.append(text)

    # Add reference lines
    ax.axvline(
      x=0 if min(data[coef]) < 0 else 1,
      linestyle='--',
      color='black',
      alpha=0.6,
    )
    ax.axhline(
      y=-np.log10(pthresh),
      linestyle='--',
      color='black',
      alpha=0.6,
    )

    # Adjust text positions to avoid overlap
    if top_n and texts:
      adjust_text(
        texts=texts,
        ax=ax,
        arrowprops=dict(arrowstyle='-', color='black', alpha=0.6),
        force_explode=5,
      )

    # Set subplot title and formatting
    ax.set_title(
      label=f'{strata[0].upper() + strata[1:].replace("_", " ")}: {stratum}',
      fontsize=12,
      fontweight='bold',
    )
    ax.grid(visible=True, alpha=0.3)
    ax.set_ylabel(ylabel='-log10(P-value)', fontsize=10)

    # Only show legend on first subplot
    if i == 0:
      ax.legend(loc='upper center', fontsize=9, framealpha=1)

  # Set common x-label
  axes[-1].set_xlabel(xlabel=xlab, fontsize=10)

  # Adjust layout
  plt.tight_layout()

  return fig
