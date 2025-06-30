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

def plot_simulated_close_price(simulation_steps:np.ndarray,predictions:np.ndarray,true_price_series:pd.Series,display_count:int = DEFAULT_NUMBER_DISPLAYED,show_legend:bool=True):
    if not (isinstance(display_count,int) and display_count > 0):
        raise ValueError("Display count is the number of simulations to plot alongside real data. Must be an integer greater than 0.")
    if not (isinstance(simulation_steps,np.ndarray) and isinstance(predictions,np.ndarray) and simulation_steps.shape[0] == predictions.shape[1]):
        raise ValueError("simulation_steps (time values) and predictions (simulated time series) must be ndarrays. The length of time_series must match the rows of predictions")
    if not (simulation_steps.ndim == 1 and predictions.ndim == 2):
        raise ValueError("simulation_steps (time values) and predictions (simulated time series) must be 1D and 2D, respectively")
    if not (isinstance(true_price_series,pd.Series) and true_price_series.shape[0] == simulation_steps.shape[0]):
        raise ValueError("real time series must have the same shape as the simulation_steps")
    if display_count > predictions.shape[1]:
        warnings.warn(f"You cannot display {display_count} simulations if predictions only have {predictions.shape[1]}. Resetting to default: {DEFAULT_NUMBER_DISPLAYED}.",category=UserWarning)
        display_count = DEFAULT_NUMBER_DISPLAYED
    if not isinstance(show_legend,bool):
        warnings.warn(f"show_legend must be a boolean. Received: {show_legend}",category=UserWarning)
        show_legend = True

    colors = _get_colors(display_count)

    for i in range(display_count):
        plt.plot(simulation_steps, predictions[i], label=f'GBM Simulation {i+1}',alpha=0.5,color=colors[i])
    plt.plot(simulation_steps, true_price_series, label=f'Real {true_price_series.name}',linewidth=3,color='red')
    plt.title(f'Real {true_price_series.name} vs. Simulated GBM Paths')
    plt.xlabel('Time (Years)')
    plt.ylabel('Price')
    plt.tight_layout()
    if show_legend:
        plt.legend()
    plt.show()

