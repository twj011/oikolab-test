"""
Weather Data Downloader for EnergyPlus Simulation.

Downloads weather data from Oikolab API and saves to CSV format.
The data can then be converted to EPW format using epw_converter.py.

Usage:
    python oikolab.py --location "Shanghai" --start 2024-01-01 --end 2024-12-31
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import io


# API Configuration
API_URL = 'https://api.oikolab.com/weather'
API_KEY = '2e7fc493ee774cfa8aadf1c41b6735d1'  # Your API key

# Required parameters for EnergyPlus simulation
REQUIRED_PARAMETERS = [
    # Temperature & Humidity (Core)
    'temperature',                      # Dry Bulb Temperature [°C]
    'dewpoint_temperature',             # Dew Point Temperature [°C]
    'relative_humidity',                # Relative Humidity [%]
    
    # Pressure
    'surface_pressure',                 # Atmospheric Pressure [Pa]
    
    # Solar Radiation (Critical for PV simulation)
    'direct_normal_solar_radiation',    # DNI [W/m²]
    'surface_diffuse_solar_radiation',  # DHI [W/m²]
    'surface_solar_radiation',          # GHI [W/m²] - for validation
    'surface_thermal_radiation',        # Infrared [W/m²] - for sky temperature
    
    # Wind (10m height, EnergyPlus will adjust)
    'wind_speed',                       # Wind Speed [m/s]
    'wind_direction',                   # Wind Direction [degrees]
    
    # Other
    'total_cloud_cover',                # Cloud Cover [fraction 0-1]
    'total_precipitation',              # Precipitation [mm]
    'snow_depth',                       # Snow Depth [m]
]


def download_weather_data(
    location: str,
    start_date: str,
    end_date: str,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Download weather data from Oikolab API.
    
    Args:
        location: Location string (e.g., "Shanghai, China")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_path: Optional path to save CSV file
        
    Returns:
        DataFrame with weather data
    """
    print(f"Downloading weather data for {location}...")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Parameters: {len(REQUIRED_PARAMETERS)} variables")
    
    # Make API request
    response = requests.get(
        API_URL,
        params={
            'param': REQUIRED_PARAMETERS,
            'location': location,
            'start': start_date,
            'end': end_date,
            'freq': 'H',  # Hourly data
            'resample_method': 'mean',
            'format': 'csv',
        },
        headers={'api-key': API_KEY}
    )
    
    # Check response
    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code}\n{response.text}")
    
    # Parse CSV response
    df = pd.read_csv(io.StringIO(response.text))
    
    # Normalize column names: "temperature (degC)" -> "temperature"
    df = normalize_column_names(df)
    
    print(f"  Downloaded {len(df)} hourly records")
    print(f"  Columns: {list(df.columns)}")
    
    # Save to file if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
    
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Oikolab column names by removing unit suffixes.
    
    Example: "temperature (degC)" -> "temperature"
    """
    rename_map = {}
    for col in df.columns:
        # Extract base name before the unit in parentheses
        if ' (' in col:
            base_name = col.split(' (')[0]
            rename_map[col] = base_name
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate downloaded weather data for EnergyPlus compatibility.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
    }
    
    # Check for required columns
    required_cols = ['temperature', 'dewpoint_temperature', 'surface_pressure',
                     'direct_normal_solar_radiation', 'surface_diffuse_solar_radiation',
                     'wind_speed', 'wind_direction']
    
    for col in required_cols:
        if col not in df.columns:
            results['errors'].append(f"Missing required column: {col}")
            results['valid'] = False
    
    # Check for missing values
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            pct = missing / len(df) * 100
            if pct > 10:
                results['warnings'].append(f"{col}: {missing} missing values ({pct:.1f}%)")
    
    # Check value ranges
    if 'temperature' in df.columns:
        if df['temperature'].min() < -50 or df['temperature'].max() > 60:
            results['warnings'].append(f"Temperature range unusual: {df['temperature'].min():.1f} to {df['temperature'].max():.1f} °C")
    
    if 'direct_normal_solar_radiation' in df.columns:
        if df['direct_normal_solar_radiation'].max() > 1400:
            results['warnings'].append(f"DNI exceeds typical max: {df['direct_normal_solar_radiation'].max():.1f} W/m²")
    
    return results


def print_data_summary(df: pd.DataFrame):
    """Print summary statistics of weather data."""
    print("\n" + "=" * 60)
    print("WEATHER DATA SUMMARY")
    print("=" * 60)
    
    # Time range
    time_col = _find_time_column(df)
    if time_col:
        print(f"\nTime Range: {df[time_col].iloc[0]} to {df[time_col].iloc[-1]}")
    
    print(f"Total Records: {len(df)}")
    
    # Key statistics
    stats_cols = [
        ('temperature', '°C', 'Dry Bulb Temperature'),
        ('dewpoint_temperature', '°C', 'Dew Point Temperature'),
        ('relative_humidity', '%', 'Relative Humidity'),
        ('direct_normal_solar_radiation', 'W/m²', 'Direct Normal Irradiance'),
        ('surface_diffuse_solar_radiation', 'W/m²', 'Diffuse Horizontal Irradiance'),
        ('wind_speed', 'm/s', 'Wind Speed'),
    ]
    
    print("\nKey Statistics:")
    print("-" * 60)
    print(f"{'Parameter':<30} {'Min':>10} {'Mean':>10} {'Max':>10}")
    print("-" * 60)
    
    for col, unit, name in stats_cols:
        if col in df.columns:
            print(f"{name:<30} {df[col].min():>10.1f} {df[col].mean():>10.1f} {df[col].max():>10.1f} {unit}")


def _find_time_column(df: pd.DataFrame) -> str:
    """Find the datetime column in DataFrame."""
    for col in ['datetime', 'time', 'timestamp', 'date']:
        if col in df.columns:
            return col
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download weather data from Oikolab for EnergyPlus simulation"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="Shanghai, China",
        help="Location string (default: Shanghai, China)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-01",
        help="Start date YYYY-MM-DD (default: 2024-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="End date YYYY-MM-DD (default: 2024-12-31)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/weather/oikolab_raw.csv",
        help="Output CSV file path",
    )
    args = parser.parse_args()
    
    # Download data
    df = download_weather_data(
        location=args.location,
        start_date=args.start,
        end_date=args.end,
        output_path=args.output,
    )
    
    # Validate
    validation = validate_data(df)
    if validation['errors']:
        print("\n⚠️ VALIDATION ERRORS:")
        for err in validation['errors']:
            print(f"  - {err}")
    if validation['warnings']:
        print("\n⚠️ WARNINGS:")
        for warn in validation['warnings']:
            print(f"  - {warn}")
    
    # Print summary
    print_data_summary(df)
    
    print("\n✅ Download complete!")
    print(f"Next step: Convert to EPW format using epw_converter.py")


if __name__ == "__main__":
    main()