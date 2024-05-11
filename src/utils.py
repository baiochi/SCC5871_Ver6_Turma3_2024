import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import DataFrame
from scipy.stats import normaltest
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

from src.palette_urban import (generate_shades, BAR, BAR_EDGE,
                               YELLOW, RED, OCEAN, CYAN, MAGENTA)

rmse_scorer = make_scorer(lambda true, pred: np.sqrt(mean_squared_error(true, pred)),
                          greater_is_better=False)


def numerical_plot(df: DataFrame,
                   column: str,
                   target_column: str = None):
    """
    Generate multiple plots for the given DataFrame and column.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - column (str): The column name to plot.
    - target_column (str, optional): The target column for scatter plot. Default is None.

    Returns:
    None
    """

    ncols = 3 if target_column else 2
    width_ratios = [1, 3, 2] if target_column else [1, 2]

    _, ax = plt.subplots(figsize=(14, 5), ncols=ncols, width_ratios=width_ratios)
    sns.boxplot(data=df, y=column, ax=ax[0],
                saturation=0.9, gap=.5,
                flierprops=dict(marker='o', markersize=2),
                color=BAR[2][1])
    title_text = ''.join(column.split('_')).title()
    ax[0].set_title(f'{title_text} Boxplot', loc='left', pad=20, size=14)
    ax[0].set_xlabel(None)
    ax[0].set_ylabel('Value')
    # Histogram
    sns.histplot(data=df, x=column, bins=20, ax=ax[1], color=BAR[2][0])
    # Set bars edge colors
    for rect in ax[1].patches:
        rect.set_edgecolor(BAR_EDGE[2][0])
        rect.set_linewidth(1)
    title_text = ''.join(column.split('_')).title()
    ax[1].set_title(f'{title_text} Histogram', loc='left', pad=20, size=14)
    ax[1].set_xlabel('Value')
    ax[1].set_ylabel('Count')

    # Scatter plot against target column
    if target_column:
        sns.scatterplot(data=df, x=column, y=target_column, ax=ax[2],
                        color=BAR[4][1], markers='o', size=1,
                        legend=False, alpha=0.7)
        title_text = ''.join(column.split('_')).title()
        title_text_target = ''.join(target_column.split('_')).title()
        ax[2].set_title(f'{title_text} vs {title_text_target}', loc='left', pad=20, size=14)
        ax[2].set_xlabel(column)
        ax[2].set_ylabel(target_column)

        plt.tight_layout()
        plt.show()


def categorical_plot(df,
                     column: str,
                     target_column: str = None,
                     orient: str = 'x',
                     width=12):
    """
    Generate a categorical plot for a given column in a DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - column (str): The column name to plot.
    - target_column (str, optional): The target column for scatter plot. Default is None.
    - orient (str, optional): The orientation of the plot. Default is 'x'.
    - width (int, optional): The width of the plot. Default is 12.

    Returns:
    None
    """

    try:
        pallete = BAR[df[column].nunique()]
    except KeyError:
        pallete = 'viridis'

    ncols = 2 if target_column else 1
    # width_ratios = [1, 3, 2] if target_column else [1, 2]

    # Countplot
    _, ax = plt.subplots(figsize=(width, 4), ncols=ncols)
    sns.countplot(data=df,
                  x=column if orient == 'x' else None,
                  y=column if orient == 'y' else None,
                  hue=column,
                  palette=pallete,
                  order=df[column].value_counts().index, gap=.3,
                  saturation=0.9, legend=False, ax=ax[0])
    title_text = ''.join(column.split('_')).title()
    ax[0].set_title(f'{title_text} Distribution', loc='left', pad=20, size=14)
    ax[0].set_xlabel('Count')
    ax[0].set_ylabel(None)

    # Boxplot correlation
    if target_column:
        sns.boxplot(data=df,
                    x=column,
                    y=target_column,
                    hue=column,
                    palette=pallete,
                    legend=False,
                    flierprops=dict(marker='o', markersize=2), ax=ax[1])
        if orient == 'y':
            # Rotate x tick labels
            ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')

        title_text = ''.join(column.split('_')).title()
        title_text_target = ''.join(target_column.split('_')).title()
        ax[1].set_title(f'{title_text} distribution across {title_text_target}', loc='left', pad=20, size=14)
        ax[1].set_xlabel(column)
        ax[1].set_ylabel(target_column)
        plt.show()


def plot_heatmap(df, target):
    numerical_cols = df.select_dtypes(include=np.number).columns
    corr_data = df[numerical_cols].corr()
    # Remove upper triangle
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    corr_data = corr_data.mask(mask)

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title(f'Correlation heatmap with target {target}', pad=20, loc='left', size=24)
    plt.show()


def get_metrics(model, X, y, cv=False, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = pd.DataFrame({
        'Train': [
            mean_absolute_error(y_train, y_pred_train),
            mean_squared_error(y_train, y_pred_train),
            np.sqrt(mean_squared_error(y_train, y_pred_train)),
            r2_score(y_train, y_pred_train),
        ],
        'Test': [
            mean_absolute_error(y_test, y_pred_test),
            mean_squared_error(y_test, y_pred_test),
            np.sqrt(mean_squared_error(y_test, y_pred_test)),
            r2_score(y_test, y_pred_test),
        ]
    }, index=['MAE', 'MSE', 'RMSE', 'R2'])

    if cv:
        metrics.loc['R2_CV'] = cross_val_score(model, X, y, cv=5).mean()
        metrics.loc['RMSE_CV'] = -cross_val_score(model, X, y, cv=5, scoring=rmse_scorer).mean()

    return metrics


def get_feature_importance(model):
    coefficients = model.named_steps['model'].coef_

    # Iterate over each step in the pipeline
    all_feature_names = []
    for name, step in model.steps:
        if isinstance(step, sklearn.compose.ColumnTransformer):
            for transformer_name, transformer in step.named_transformers_.items():
                try:
                    feature_names = transformer.get_feature_names_out()
                except AttributeError:
                    feature_names = transformer.get_feature_names_in()
                all_feature_names.extend(feature_names)
        elif isinstance(step, sklearn.preprocessing.PolynomialFeatures):
            all_feature_names = step.get_feature_names_out(all_feature_names)

    return pd.Series(coefficients, index=all_feature_names)


def plot_regression(y_test, y_pred_test, scores, order, ax):
    sns.regplot(x=y_test,
                y=y_pred_test,
                order=order,
                scatter_kws={"color": CYAN},
                line_kws={"color": OCEAN},
                marker='.',
                ax=ax)
    ax.set_title('Predicted vs True', loc='left', pad=20, size=18)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.legend([f'R2_CV: {scores[0]:.2f}\nRMSE_CV: {scores[1]:.2f}'], loc='lower right')


def plot_redisuals(resid, ax):
    _, p_value = normaltest(resid)
    sns.histplot(resid, kde=True, color=YELLOW, alpha=.8, ax=ax)
    ax.set_title('Residuals Distribution', loc='left', pad=20, size=18)
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')
    if p_value < 0.05:
        ax.annotate(f'Normality Rejected\n(p-value={p_value:.5f})',
                    xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', color=RED)
    else:
        ax.annotate(f'Normality Accepted\n(p-value={p_value:.5f})',
                    xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', color=OCEAN)


def plot_feature_importance(feature_importance, ax):
    # Avoid UserWarning from sns.barplot
    palette_size = len(feature_importance.index)
    palette = generate_shades(CYAN, RED, len(feature_importance))[:palette_size]

    sns.barplot(x=feature_importance, y=feature_importance.index,
                hue=feature_importance.index.str.split('_').str[0],
                palette=palette,
                legend=False, ax=ax)
    ax.set_title('Feature Importance', loc='left', pad=20, size=18)
    ax.set_xlabel('Coefficient')
    ax.set_ylabel(None)


def plot_regression_results(
        y_test,
        y_pred,
        scores: tuple[float, float],
        feature_importance,
        order=1,
        fig_width=15,
        fig_height=5
):

    if len(feature_importance) < 12:
        _, ax = plt.subplots(1, 3, figsize=(fig_width, fig_height))
        plot_regression(y_test, y_pred,scores, order, ax[0])
        plot_redisuals(y_test - y_pred, ax[1])
        plot_feature_importance(feature_importance, ax[2])
    else:
        # Plot features importance isolated if number of features is too high
        _, ax = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        plot_regression(y_test, y_pred,scores, order, ax[0])
        plot_redisuals(y_test - y_pred, ax[1])
        _, ax = plt.subplots(figsize=(6, 12))
        plot_feature_importance(feature_importance, ax)

    plt.tight_layout()
    plt.show()


def metrics_report(
        model_metrics: list[DataFrame],
        model_names: list[str],
        plot_metrics: list[str] = None,
        fig_width=10,
        fig_height=4
):
    assert len(model_metrics) == len(model_names)

    # Melt dataframe to plot metrics
    metrics = pd.concat([
        df
        .rename_axis('metrics')
        .reset_index()
        .melt(id_vars='metrics', var_name='dataset', value_name='value')
        .assign(model=name)
        for df, name in zip(model_metrics, model_names)
    ]).reset_index(drop=True)

    test_mask = metrics['dataset'] == 'Test'
    r2_mask = metrics['metrics'] == 'R2_CV'
    rmse_mask = metrics['metrics'] == 'RMSE_CV'
    max_r2_index = metrics.loc[test_mask & r2_mask, 'value'].idxmax()
    lower_rmse_index = metrics.loc[test_mask & rmse_mask, 'value'].idxmin()

    print('Model with higher R²-score (avg. CV):',
          metrics.loc[max_r2_index, 'model'],
          f'({metrics.loc[max_r2_index, "value"]:.5f})')
    print('Model with lower RMSE-score (avg. CV):',
          metrics.loc[lower_rmse_index, 'model'],
          f'({metrics.loc[lower_rmse_index, "value"]:.5f})')

    # Plot metrics
    if plot_metrics:
        ncols = 2 if len(plot_metrics) > 1 else 1
        nrows = 1 if len(plot_metrics) == 1 else len(plot_metrics) // 2
        _, ax = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))

        for i, metric in enumerate(plot_metrics):
            sns.barplot(data=metrics[metrics['metrics'] == metric],
                        x='dataset', y='value', hue='model',
                        gap=0.5, legend=False if i else True,
                        palette=BAR[len(model_metrics)],
                        ax=ax.flatten()[i])
            ax.flatten()[i].set_title(metric, loc='left', fontsize=16, pad=20)
            ax.flatten()[i].set_ylabel('')
            ax.flatten()[i].set_xlabel('')
        # Model legend box
        ax.flatten()[0].legend(title='Model', loc='upper left', bbox_to_anchor=(1, .5))

        # Main title
        plt.suptitle('Metrics comparison across models', fontsize=24)
        plt.tight_layout()
        plt.show()
