import numpy as np
import pandas as pd
import holoviews as hv
from scipy.signal import savgol_filter

def detect_water_temp(temp,time,window_size=6, only_water_val=True):

    # Classify based on moving variance
    moving_avg = pd.Series(temp).rolling(window=window_size, min_periods=1).mean().to_numpy()
    moving_var = pd.Series(temp).rolling(window=window_size, min_periods=1).var().to_numpy()

    var_threshold = np.nanmedian(moving_var)*4
    mean_threshold = np.nanmedian(moving_avg)

    ithreshold = np.where(moving_var > var_threshold)[0][0]
    #print(ithreshold)

    state = np.full(temp.shape, 'air')
    state[ithreshold:] = np.where((moving_var[ithreshold:] < var_threshold) ,'wat','air')
    #print(state)

    #state = np.where((moving_var < var_threshold) & (moving_avg < mean_threshold), 'water', 'air')

    # Forward fill and backward to smooth
    state_filled = pd.Series(state).ffill().bfill().to_numpy()
    water_data = temp[state_filled == 'wat']
    air_data = temp[state_filled == 'air']
    water_time = time[state_filled == 'wat']
    air_time = time[state_filled == 'air']

    if only_water_val:
      return np.nanmean(water_data)
    else:
      return water_data, water_time, moving_var, var_threshold

def get_data_from_temp_sensors(filepath, team_name='Escuela Salle', location='IEO', lat= None, lon= None, ):

    data=pd.read_csv(filepath)
    time=pd.to_datetime(data.iloc[21::].iloc[:, 0]).to_numpy()
    temp=data.iloc[21::].iloc[:,1].to_numpy().astype('float')

    water_data, water_time, moving_var, var_threshold = detect_water_temp (temp, time, window_size=6, only_water_val=False)
    water_time=water_time[0]
    water_temp=np.nanmean(water_data)
    time_str = np.datetime_as_string(water_time, unit='D')
    month = water_time.astype('datetime64[M]').astype(int) % 12 + 1  # Months (1 to 12)
    day = (water_time - water_time.astype('datetime64[M]')).astype('timedelta64[D]').astype(int) + 1  # Days
    days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    fractional_week = (sum(days_in_months[:month-1]) + day - 1) / 7
    
    if not lat:
      lat=float(data.iloc[16][1])
      lon=float(data.iloc[17][1])


    df = pd.DataFrame([{
    'fractional_week': fractional_week,  # Approx. week 52 for Dec 24
    'Temperature': water_temp,
    'Date': time_str,
    'Latitude': lat,
    'Longitude': lon,
    'Team': team_name,
    'Location':location}])

    return df


def plot_climatological_year(file_path):
    
    # Load the dataset
    df = pd.read_excel(file_path, header=0)
    df = df.rename(columns={'año': 'year', 'mes': 'month', 'dia': 'day', 'temperatura agua': 'temperatura'})

    # Create a Date column
    df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])

    # Group by month for climatological statistics
    grouped = df.groupby('month')['temperatura']
    q1 = grouped.quantile(0.25)
    q3 = grouped.quantile(0.75)
    iqr = q3 - q1
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr
    medians = grouped.median()

    days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Function to calculate fractional week
    def calculate_fractional_week(row):
    # Sum up the days for all previous months
      days_before = sum(days_in_months[:row['month'] - 1])
    # Add the current day and convert to fractional week
      return (days_before + row['day'] - 1) / 7

# Apply the function to calculate fractional weeks
    df['fractional_week'] = df.apply(calculate_fractional_week, axis=1)


    # Weekly indices for interpolation
    weekly_indices = np.linspace(1, 52, 52)
    weeks_per_month = [4.345] * 12
    cumulative_weeks = np.cumsum([0] + weeks_per_month)

    # Interpolate whiskers and medians to weekly scale
    weekly_medians = np.interp(weekly_indices, cumulative_weeks[1:], medians)
    weekly_lower = np.interp(weekly_indices, cumulative_weeks[1:], lower_whisker)
    weekly_upper = np.interp(weekly_indices, cumulative_weeks[1:], upper_whisker)

    # Smooth whiskers and medians
    smoothed_medians = savgol_filter(weekly_medians, 7, 2)
    smoothed_lower = savgol_filter(weekly_lower, 7, 2)
    smoothed_upper = savgol_filter(weekly_upper, 7, 2)

    # Create the climatological year plot using Holoviews (with vdims)
    climato_plot = hv.Area(
        (weekly_indices, smoothed_lower, smoothed_upper),
        vdims=['y', 'y2'],
        label="IQR-based Range"
    ).opts(
        color='lightblue',
        alpha=0.5,
        title='Climatological Year Temperature Range',
        xlabel='Climatological Year',
        ylabel='Temperature',
        width=800,
        height=400
        )

    # Add the smoothed median line (in weekly scale)
    climato_plot_median = hv.Curve(
        (weekly_indices, smoothed_medians),
        label='Median'
    ).opts(
        color='blue',
        line_width=2
        )

    month_boundaries = [
      0,    # January
      4.345, # February
      8.69,  # March
      13.035, # April
      17.38,  # May
      21.725, # June
      26.07,  # July
      30.415, # August
      34.76,  # September
      39.105, # October
      43.45,  # November
      47.795  # December
    ]

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create the combined plot
    combined_plot = climato_plot * climato_plot_median

# Set x-ticks and labels for months
    combined_plot.opts(
      xticks=[(i, label) for i, label in zip(month_boundaries, month_labels)],
      show_grid=True
    )

    # Show the combined plot
    return combined_plot

def plot_obs(df):
    scatter_plot = hv.Scatter(
      df,
      kdims=['fractional_week'],
      vdims=['Temperature', 'Date', 'Team','Location']
    ).opts(
      tools=['hover'],
      width=800,
      height=400,
      title='Interactive Climatological Data',
      color='red',
      size=8,
      xlabel='Climatological Year (Weeks)',
      ylabel='Temperature',
      legend_position='top_left'
      )
    return scatter_plot

def get_data_from_temp_sensors_full(filepath, team_name='Escuela Salle', location='IEO', lat=None, lon=None):
    # Read the data
    data = pd.read_csv(filepath)
    
    # Extract time and temperature data
    time = pd.to_datetime(data.iloc[21::].iloc[:, 0]).to_numpy()
    temp = data.iloc[21::].iloc[:, 1].to_numpy().astype('float')
    
    # Process the data
    water_data, water_time, moving_var, var_threshold = detect_water_temp(
        temp, time, window_size=6, only_water_val=False
    )
    
    # Convert times to string format with minute precision
    time_str = np.array([t.astype('datetime64[m]').astype(str) for t in water_time])
    
    # Get latitude and longitude if not provided
    if not lat:
        lat = float(data.iloc[16][1])
        lon = float(data.iloc[17][1])
    
    # Create the DataFrame in "tidy" format
    df_full = pd.DataFrame({
        'Temperature': water_data,       # Expand temperatures
        'Date': water_time,                # Expand corresponding times
        'Latitude': [lat] * len(water_data),    # Repeat metadata for each row
        'Longitude': [lon] * len(water_data),
        'Team': [team_name] * len(water_data),
        'Location': [location] * len(water_data),
    })

    return df_full


