import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

DEFAULT_NUMBER_DISPLAYED = 5

def _get_colors(display_count:int):
    return plt.cm.viridis(np.linspace(0, 1, display_count))

def _single_moment_set(df:pd.DataFrame, column:str) -> pd.DataFrame:
    """
    Creates a DataFrame for the four moments of the desired column

    Parameters:
    df (DataFrame): DataFrame of numerical values (at least in desired column)
    column (str): Column in df that contains numerical rows

    Returns:
    DataFrame: A new DataFrame with the original column and new rows for each moment (mean, std, skew, kurt)
    """

    column_data = df[column]
    return pd.DataFrame(
        data=[column_data.mean(),column_data.std(),column_data.skew(),column_data.kurtosis()],
        index=["mean","std","skew","kurt"],
        columns=[column]
        )

def _moments_df(df:pd.DataFrame) -> pd.DataFrame:
    """
    Creates a dataframe with all four moments of the provided dataframe

    Parameters:
    df (DataFrame): DataFrame of numerical values

    Returns:
    DataFrame: same columns as df with four indexes representing the four moments of the provided columns
    """

    columns = df.columns
    new_df = None
    for column in columns:
        if new_df is None:
            new_df = _single_moment_set(df,column)
        else:
            new_df = new_df.join(_single_moment_set(df,column))
    
    return new_df

def plot_simulated_close_price(predictions:pd.DataFrame,true_prices:pd.DataFrame,display_count:int = DEFAULT_NUMBER_DISPLAYED,date_or_time:bool=False,show_legend:bool=True):
    if not (isinstance(display_count,int) and display_count > 0):
        raise ValueError("Display count is the number of simulations to plot alongside real data. Must be an integer greater than 0.")
    if not (isinstance(predictions,pd.DataFrame) and isinstance(true_prices,pd.DataFrame) and true_prices.shape[0] == predictions.shape[0] and true_prices.shape[0] > 0):
        raise ValueError("predictions and true_price must be the outputs of gbm_price_sim, i.e, two dataFrames with the same number of time-points.")
    if not (true_prices.index.symmetric_difference(predictions.index).shape == (0,) and true_prices.shape[1] > 1 and predictions.shape[1] > 1):
        raise ValueError("predictions and true_price must have the same indexes. They should also have at least one column")
    if display_count > predictions.shape[1]:
        warnings.warn(f"You cannot display {display_count} simulations if predictions only have {predictions.shape[1]}. Resetting to default: {DEFAULT_NUMBER_DISPLAYED}.",category=UserWarning)
        display_count = DEFAULT_NUMBER_DISPLAYED
    if not isinstance(show_legend,bool):
        warnings.warn(f"show_legend must be a boolean. Received: {show_legend}. Default reset to True",category=UserWarning)
        show_legend = True
    if not isinstance(date_or_time,bool):
        warnings.warn(f"date_or_time must be a boolean. Received: {date_or_time}. Default reset to True",category=UserWarning)
        date_or_time = True


    colors = _get_colors(display_count)
    all_times = predictions.index
    if date_or_time:
        x_values = all_times
    else:
        T = (all_times[-1] - all_times[0]).days / 364.25
        x_values = np.linspace(0,T,int(predictions.shape[0]))

    for i in range(display_count):
        plt.plot(x_values, predictions.iloc[:,i], label=f'GBM Simulation {i+1}',alpha=0.5,color=colors[i])
    plt.plot(x_values, true_prices.iloc[:,0], label=f'Real {true_prices.columns[0]}',linewidth=3,color='red')
    plt.title(f'Real {true_prices.columns[0]} vs. Simulated GBM Paths')
    x_label = 'Time (Years)' if not date_or_time else 'Dates'
    plt.xlabel(x_label)
    plt.ylabel('Price')
    plt.tight_layout()
    if show_legend:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_simulated_log_return(log_predictions:pd.DataFrame,true_prices:pd.DataFrame,display_count:int=DEFAULT_NUMBER_DISPLAYED,show_legend:bool=True):
    if not (isinstance(display_count,int) and display_count > 0):
        raise ValueError("Display count is the number of simulations to plot alongside real data. Must be an integer greater than 0.")
    if not (isinstance(log_predictions,pd.DataFrame) and isinstance(true_prices,pd.DataFrame) and true_prices.shape[0] == log_predictions.shape[0] and true_prices.shape[0] > 0):
        raise ValueError("predictions and true_price must be the outputs of gbm_price_sim, i.e, two dataFrames with the same number of time-points.")
    if not (true_prices.index.symmetric_difference(log_predictions.index).shape == (0,) and true_prices.shape[1] > 1 and log_predictions.shape[1] > 1):
        raise ValueError("predictions and true_price must have the same indexes. They should also have at least one column")
    if display_count > log_predictions.shape[1]:
        warnings.warn(f"You cannot display {display_count} simulations if predictions only have {log_predictions.shape[1]}. Resetting to default: {DEFAULT_NUMBER_DISPLAYED}.",category=UserWarning)
        display_count = DEFAULT_NUMBER_DISPLAYED
    if not isinstance(show_legend,bool):
        warnings.warn(f"show_legend must be a boolean. Received: {show_legend}. Default reset to True",category=UserWarning)
        show_legend = True

    colors = _get_colors(display_count)
    cleaned_predications, cleaned_true = log_predictions.dropna(), true_prices.dropna()
    for i in range(display_count):
        plt.hist(cleaned_predications.dropna().iloc[:,i], label=f'GBM Simulation {i+1}',alpha=0.5,color=colors[i],density=True,bins=50)
    plt.hist(cleaned_true.dropna().iloc[:,1], label=f'Real {true_prices.columns[1]}',color='red',density=True,bins=50)
    plt.title(f'Real {true_prices.columns[1]} and GBM Simulation')
    plt.xlabel('LOG RETURNS')
    plt.ylabel('DENSITY')
    plt.tight_layout()
    if show_legend:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

