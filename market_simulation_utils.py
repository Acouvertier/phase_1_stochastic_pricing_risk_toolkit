import numpy as np
import pandas as pd
import warnings
import yfinance as yf

from collections.abc import Callable
from scipy.stats import rv_discrete, norm
from typing import Tuple, List, Dict

# Symmetric walk
_symmetric_walk = rv_discrete(a=-1,b=1,values=([-1,1],[0.5,0.5]))

def _pay_off_euro_call(price:float,strike:float) -> float:
    try:
        if not (isinstance(price,float)):
            price = float(price)
        if not (isinstance(strike,float)):
            strike = float(strike)
    except ValueError as e:
        raise ValueError(f"The provided strike or pice couldn't be type-cast to float: {e}")
    
    return max(0,price-strike)

def _pay_off_euro_put(price:float,strike:float) -> float:
    try:
        if not (isinstance(price,float)):
            price = float(price)
        if not (isinstance(strike,float)):
            strike = float(strike)
    except ValueError as e:
        raise ValueError(f"The provided strike or pice couldn't be type-cast to float: {e}")

    return max(0,strike-price)

def _pay_off_forward(price:float,strike:float) -> float:
    try:
        if not (isinstance(price,float)):
            price = float(price)
        if not (isinstance(strike,float)):
            strike = float(strike)
    except ValueError as e:
        raise ValueError(f"The provided strike or pice couldn't be type-cast to float: {e}")

    return price - strike

_PAY_OFF_DICT = {
    "euro_call": _pay_off_euro_call, 
    "euro_put": _pay_off_euro_put, 
    "forward": _pay_off_forward
    }

def _format_single_column(stock_data:pd.DataFrame,column_name:str) -> pd.DataFrame:
    just_close = stock_data[column_name]["Close"]
    just_close = just_close.rename(f"CLOSE_{column_name}")
    just_log = np.log(just_close.pct_change()+1)
    just_log = just_log.rename(f"LOG_RETURN_{column_name}")
    return pd.concat([just_close,just_log],axis=1)

def brownian_rvs(dt:float|int,M:int,steps:int) -> np.ndarray:
    if not (isinstance(dt,(float,int)) and dt > 0 and isinstance(M,int) and M > 0 and isinstance(steps,int) and steps > 0):
        raise ValueError("time step must be numeric (int or float), M and steps must be positive integers.")
    return norm.rvs(loc=0,scale=np.sqrt(dt),size=(M,steps))

def symmetric_random_walk_discrete(N:int,M:int)-> Tuple[np.ndarray,np.ndarray]:
    if not (isinstance(N,int) and isinstance(M,int)):
        raise ValueError("Symmetric Random Walk defined for integer steps (N) and realizations (M).")
    if not (M > 0 and N > 0):
        raise ValueError("Both Step Size (N) and Realizations (M) must be positive integers.")
    steps = _symmetric_walk.rvs(size=(M,N))
    loc = np.cumsum(steps,axis=1)
    loc = np.hstack([np.zeros((M,1)),loc])
    step_count = np.arange(0,N+1)
    
    return (step_count,loc)

def path_increment_discrete(l:int,k:int,paths:np.ndarray) -> Tuple[float,float]:
    if not (isinstance(l,int) and isinstance(k,int)):
        raise ValueError("Path increments must be integer values")
    elif not (isinstance(paths,np.ndarray) and paths.ndim == 2 and paths.shape[0] > 0 and paths.shape[1] > 0):
        raise ValueError("paths must be a collection of realized random processes of equal length")
    if l <= k:
        warnings.warn("l cannot be less than k, they will be flipped.",category=UserWarning)
        l, k = k, l
    increments = paths[:,l] - paths[:,k]
    mean = np.mean(increments) 
    var = np.var(increments,mean=mean)
    return (mean,var)

def quadratic_variation(k:int,path:np.ndarray) -> float:
    if not (isinstance(k,int) and k>0):
        raise ValueError("k must be a positive integer")
    elif not (isinstance(path,np.ndarray) and path.ndim == 1):
        raise ValueError("path must be the single realization of a random variable.")
    elif not (path.shape[0] >= k):
        warnings.warn(f"Requested k={k} exceeds path length={path.shape[0]}. Resetting k to full path length.", category=UserWarning)
        k = path.shape[0]
    increments = np.diff(path,n=1)
    squared_increments = np.square(increments)
    
    return np.sum(squared_increments[:k])

def approximate_brownian_motion(n:int,t:int,M:int) -> Tuple[np.ndarray,np.ndarray]:
    if not (isinstance(n,int) and isinstance(t,int) and isinstance(M,int) and n > 0 and t > 0 and M > 0):
        raise ValueError("All inputs must be positive integers")
    
    N = n*t
    step_count, loc = symmetric_random_walk_discrete(N,M)
    return step_count/n, (1/np.sqrt(n))*loc

def continuous_brownian_motion(T:float|int,intervals:int,realizations:int) -> Tuple[np.ndarray,np.ndarray]:
    if not (isinstance(intervals,int) and intervals > 0):
        raise ValueError("intervals (the number of points between 0 and T) must be a postive integer")
    if not (isinstance(T,(float,int)) and T > 0):
        raise ValueError("T (end time) must be a positive number")
    if not (isinstance(realizations,int) and realizations > 0):
        raise ValueError("realizations (number of trajectories) must be a positive integer.")
    t_space = np.linspace(0,T,num=intervals)
    dt = t_space[1] - t_space[0]
    increments = brownian_rvs(dt,realizations,intervals-1)
    paths = np.cumsum(increments,axis=1)
    paths = np.hstack([np.zeros((realizations,1)),paths])
    return t_space, paths

def geometric_brownian_motion(start_price:float|int,mean:float|int,volatility:float|int,T:float|int,steps:int,realizations:int) -> Tuple[np.ndarray,np.ndarray]:
    if not (isinstance(start_price,(float,int)) and isinstance(mean,(float,int)) and isinstance(volatility,(float,int))):
        raise ValueError("start_price, mean, and volatility must all be numeric (float or int).")
    if not (volatility > 0):
        raise ValueError("Volatility should not be negative")
    
    t_space, paths = continuous_brownian_motion(T,steps,realizations)
    vol_brownian = volatility * paths
    drift = (mean - 0.5*(volatility**2)) * t_space[np.newaxis,:]

    return t_space, start_price * np.exp(vol_brownian + drift)

def fetch_stock_data(start:str,end:str,ticker_list:List[str]):
    raw_data = yf.download(ticker_list,start=start,end=end,group_by='ticker',keepna=True,ignore_tz=True)
    if raw_data is None:
        raise ValueError("Yahoo Finance was unable to be queried. Ensure start and end are valid datetime strings (YYYY-MM-DD) annd the stock_list contains valid tickers.")
    pruned_data = []
    for ticker in ticker_list:
        pruned_data.append(_format_single_column(raw_data,ticker))

    return pd.concat(pruned_data,axis=1)

def gbm_price_sim(start:str,end:str,stock_ticker:str, realizations:int) -> Tuple[pd.DataFrame,np.ndarray,np.ndarray]:

    price_log_data = fetch_stock_data(start,end,[stock_ticker])
    close_log_return = price_log_data[f"LOG_RETURN_{stock_ticker}"]
    close_price = price_log_data[f"CLOSE_{stock_ticker}"]
    all_times = price_log_data.index
    T = (all_times[-1] - all_times[0]).days / 364.25

    initial_price = close_price.iloc[0]
    empirical_vol = np.sqrt(252) * close_log_return.std()
    empirical_mean = (1/T)*np.log(close_price.iloc[-1]/initial_price) + 0.5*empirical_vol**2

    time_steps, simulated_prices = geometric_brownian_motion(initial_price,empirical_mean,empirical_vol,T,len(all_times),realizations)

    return time_steps, pd.DataFrame(data=simulated_prices.T,index=price_log_data.index,columns=[(f"CLOSE_SIM_{i+1}").upper() for i in range(realizations)]), price_log_data

def log_returns_gbm_price_sim(simulation_df:pd.DataFrame) -> pd.DataFrame:

    log_return_cols = []
    for _, col_data in simulation_df.items():
        log_returns = np.log(col_data.pct_change() + 1)
        log_return_cols.append(log_returns.to_frame())

    log_return_df = pd.concat(log_return_cols,axis=1)
    log_return_df.columns = [f"LOG_RETURN_SIM_{i+1}" for i in range(simulation_df.shape[1])]

    return log_return_df

def no_arbitrage_probs(up_factor:float,down_factor:float,rate:float) -> Tuple[float,float]:
    try:
        if not (isinstance(up_factor,float)):
            up_factor = float(up_factor)
        if not (isinstance(down_factor,float)):
            down_factor = float(down_factor)
        if not (isinstance(rate,float)):
            rate = float(rate)
    except ValueError as e:
        raise ValueError(f"The provided factor or rate couldn't be type-cast to float: {e}")
    if not (down_factor > 0 and down_factor < 1 + rate and 1 + rate < up_factor):
        raise ValueError("No arbitrarge requires: 0 <= down_factor < 1 + rate < up_factor <= 1")
    
    p_0 = (1+rate-down_factor)/(up_factor - down_factor)

    return (p_0, 1 - p_0)

def fair_option_price_multi_period(periods:int,initial_price:float,up_factor:float,down_factor:float,rate:float,strike:float,pay_off_funct:Callable[[float,float],float]=_PAY_OFF_DICT["euro_call"]) -> Tuple[Dict, float]:
    
    p_0, q_0 = no_arbitrage_probs(up_factor,down_factor,rate)

    end_prices =  np.array([initial_price *(up_factor**(periods-i))*(down_factor**(i)) for i in range(periods+1)])
    end_values = np.array([pay_off_funct(initial_price *(up_factor**(periods-i))*(down_factor**(i)),strike) for i in range(periods+1)])

    hedge_shares = {}

    for i in reversed(range(periods)):
        hedge_shares[i] = np.divide(np.diff(end_values),np.diff(end_prices))
        end_values = (1/(1+rate))*(p_0*(end_values[:-1])+q_0*(end_values[1:]))
        end_prices = (1/up_factor)*end_prices[1:]
    
    return float(end_values[0]), hedge_shares