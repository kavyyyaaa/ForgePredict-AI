import pandas as pd
import numpy as np

def forecast_sensor_trend(historical_values, time_steps_ahead=24, interval_mins=30, fail_threshold=8.0, is_increasing=True):
    """
    Applies Double Exponential Smoothing (Holt's Linear Trend) to project sensor values.
    Also calculates the projected time (in hours) until the value crosses the fail_threshold.
    """
    y = np.array(historical_values)
    n = len(y)
    
    if n < 5:
        # Fallback if too few points
        last_val = y[-1] if n > 0 else 0.0
        forecast = np.full(time_steps_ahead, last_val)
        return forecast, None
    
    # Holt's Double Exponential Smoothing parameters
    alpha = 0.3
    beta = 0.1
    
    # Initial level and trend
    level = y[0]
    trend = y[1] - y[0]
    
    # Fit model
    for i in range(1, n):
        last_level = level
        level = alpha * y[i] + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        
    # Generate forecast
    forecast = []
    time_to_breach_steps = None
    
    for m in range(1, time_steps_ahead + 1):
        pred = level + m * trend
        # Ensure we don't return negative values for sensors that should be positive
        pred = max(0.1, pred)
        forecast.append(pred)
        
        # Check for breach
        if time_to_breach_steps is None:
            if is_increasing and pred >= fail_threshold:
                time_to_breach_steps = m
            elif not is_increasing and pred <= fail_threshold:
                time_to_breach_steps = m
                
    forecast = np.array(forecast)
    
    time_to_breach_hours = None
    if time_to_breach_steps is not None:
        time_to_breach_hours = float(time_to_breach_steps * (interval_mins / 60.0))
        
    return forecast, time_to_breach_hours
