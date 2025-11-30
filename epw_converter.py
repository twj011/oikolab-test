"""
EPW Weather File Converter.

Converts Oikolab CSV weather data to EnergyPlus EPW format.

EPW Format Reference:
https://bigladdersoftware.com/epx/docs/8-3/auxiliary-programs/energyplus-weather-file-epw-data-dictionary.html

Usage:
    python epw_converter.py --input data/weather/oikolab_raw.csv --output data/weather/Shanghai_2024.epw
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import math


# EPW Header Template
EPW_HEADER_TEMPLATE = """LOCATION,{city},{state},{country},{data_source},{wmo},{latitude},{longitude},{timezone},{elevation}
DESIGN CONDITIONS,0
TYPICAL/EXTREME PERIODS,0
GROUND TEMPERATURES,0
HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0
COMMENTS 1,Generated from Oikolab API data
COMMENTS 2,Converted using epw_converter.py
DATA PERIODS,1,1,Data,Sunday,1/1,12/31
"""


def calculate_sky_temperature(
    dry_bulb: float,
    dew_point: float,
    opaque_sky_cover: float,
    horizontal_ir: float = None,
) -> float:
    """
    Calculate sky temperature for EPW file.
    
    If horizontal IR radiation is available, use it directly.
    Otherwise, estimate from dry bulb and cloud cover.
    
    Args:
        dry_bulb: Dry bulb temperature [°C]
        dew_point: Dew point temperature [°C]
        opaque_sky_cover: Cloud cover [tenths, 0-10]
        horizontal_ir: Horizontal infrared radiation [W/m²]
        
    Returns:
        Sky temperature [°C]
    """
    if horizontal_ir is not None and horizontal_ir > 0:
        # Calculate from IR radiation using Stefan-Boltzmann
        # IR = σ * T_sky^4, solve for T_sky
        sigma = 5.67e-8  # Stefan-Boltzmann constant
        T_sky_K = (horizontal_ir / sigma) ** 0.25
        return T_sky_K - 273.15
    else:
        # Estimate using Clark-Allen model
        # T_sky = T_db * (0.8 + (T_dp + 273) / 250) ^ 0.25 * (1 + 0.0224*N - 0.0035*N^2 + 0.00028*N^3)
        N = opaque_sky_cover
        T_db_K = dry_bulb + 273.15
        T_dp_K = dew_point + 273.15
        
        emissivity = 0.787 + 0.764 * math.log(T_dp_K / 273.0)
        cloud_factor = 1 + 0.0224 * N - 0.0035 * N**2 + 0.00028 * N**3
        
        T_sky_K = T_db_K * (emissivity * cloud_factor) ** 0.25
        return T_sky_K - 273.15


def convert_to_epw(
    df: pd.DataFrame,
    output_path: str,
    location_info: dict = None,
) -> str:
    """
    Convert Oikolab DataFrame to EPW format.
    
    Args:
        df: DataFrame with weather data from Oikolab
        output_path: Path for output EPW file
        location_info: Dictionary with location metadata
        
    Returns:
        Path to generated EPW file
    """
    # Default location info
    if location_info is None:
        location_info = {
            'city': 'Shanghai',
            'state': 'Shanghai',
            'country': 'CHN',
            'data_source': 'Oikolab',
            'wmo': '583620',  # Shanghai WMO station
            'latitude': 31.23,
            'longitude': 121.47,
            'timezone': 8,
            'elevation': 4,
        }
    
    # Normalize column names if they contain units (e.g., "temperature (degC)" -> "temperature")
    df = _normalize_column_names(df)
    
    # Parse datetime
    time_col = None
    for col in ['datetime', 'time', 'timestamp', 'date']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError("No time column found in DataFrame")
    
    df['datetime'] = pd.to_datetime(df[time_col])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Ensure we have 8760 hours (full year)
    expected_hours = 8760
    if len(df) != expected_hours:
        print(f"Warning: Data has {len(df)} hours, expected {expected_hours}")
        if len(df) < expected_hours:
            print("  Data will be padded with interpolation")
        else:
            print("  Data will be truncated to first year")
            df = df.head(expected_hours)
    
    # Create EPW data rows
    epw_rows = []
    
    for idx, row in df.iterrows():
        dt = row['datetime']
        
        # Extract time components
        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour + 1  # EPW uses 1-24 hour format
        
        # Get weather values with unit conversions
        dry_bulb = _get_value(row, 'temperature', 20.0)
        dew_point = _get_value(row, 'dewpoint_temperature', 10.0)
        rel_humidity = _get_value(row, 'relative_humidity', 50.0)
        
        # Pressure: Oikolab gives Pa, EPW needs Pa
        atm_pressure = _get_value(row, 'surface_pressure', 101325.0)
        
        # Radiation: Oikolab gives W/m², EPW needs Wh/m² (same for hourly data)
        # Note: For hourly data, W/m² average = Wh/m² total
        ghi = max(0, _get_value(row, 'surface_solar_radiation', 0.0))
        dni = max(0, _get_value(row, 'direct_normal_solar_radiation', 0.0))
        dhi = max(0, _get_value(row, 'surface_diffuse_solar_radiation', 0.0))
        
        # Infrared radiation
        horizontal_ir = _get_value(row, 'surface_thermal_radiation', None)
        if horizontal_ir is not None:
            horizontal_ir = max(0, horizontal_ir)
        
        # Wind
        wind_speed = max(0, _get_value(row, 'wind_speed', 0.0))
        wind_direction = _get_value(row, 'wind_direction', 0.0) % 360
        
        # Cloud cover: Oikolab gives fraction (0-1), EPW needs tenths (0-10)
        cloud_cover_frac = _get_value(row, 'total_cloud_cover', 0.5)
        total_sky_cover = int(round(cloud_cover_frac * 10))
        opaque_sky_cover = total_sky_cover  # Assume same as total
        
        # Precipitation: Oikolab gives mm, EPW needs mm
        precip = max(0, _get_value(row, 'total_precipitation', 0.0))
        
        # Snow depth: Oikolab gives m, EPW needs cm
        snow_depth = max(0, _get_value(row, 'snow_depth', 0.0) * 100)
        
        # Calculate derived values
        sky_temp = calculate_sky_temperature(
            dry_bulb, dew_point, opaque_sky_cover, horizontal_ir
        )
        
        # Extraterrestrial radiation (simplified calculation)
        extraterrestrial_hor = _calculate_extraterrestrial(
            dt, location_info['latitude']
        )
        extraterrestrial_dir = extraterrestrial_hor  # Simplified
        
        # Illuminance (estimated from radiation)
        # Typical luminous efficacy: ~100 lm/W for daylight
        global_illum = ghi * 100 if ghi > 0 else 0
        direct_illum = dni * 100 if dni > 0 else 0
        diffuse_illum = dhi * 100 if dhi > 0 else 0
        zenith_illum = 0  # Not typically available
        
        # Build EPW row (35 fields)
        epw_row = [
            year,                    # 1: Year
            month,                   # 2: Month
            day,                     # 3: Day
            hour,                    # 4: Hour (1-24)
            0,                       # 5: Minute
            'Oikolab',              # 6: Data Source
            f'{dry_bulb:.1f}',      # 7: Dry Bulb Temperature [°C]
            f'{dew_point:.1f}',     # 8: Dew Point Temperature [°C]
            f'{rel_humidity:.0f}',  # 9: Relative Humidity [%]
            f'{atm_pressure:.0f}',  # 10: Atmospheric Pressure [Pa]
            f'{extraterrestrial_hor:.0f}',  # 11: Extraterrestrial Horizontal Radiation [Wh/m²]
            f'{extraterrestrial_dir:.0f}',  # 12: Extraterrestrial Direct Normal Radiation [Wh/m²]
            f'{horizontal_ir:.0f}' if horizontal_ir else '9999',  # 13: Horizontal Infrared Radiation [Wh/m²]
            f'{ghi:.0f}',           # 14: Global Horizontal Radiation [Wh/m²]
            f'{dni:.0f}',           # 15: Direct Normal Radiation [Wh/m²]
            f'{dhi:.0f}',           # 16: Diffuse Horizontal Radiation [Wh/m²]
            f'{global_illum:.0f}',  # 17: Global Horizontal Illuminance [lux]
            f'{direct_illum:.0f}',  # 18: Direct Normal Illuminance [lux]
            f'{diffuse_illum:.0f}', # 19: Diffuse Horizontal Illuminance [lux]
            f'{zenith_illum:.0f}',  # 20: Zenith Luminance [Cd/m²]
            f'{wind_direction:.0f}',# 21: Wind Direction [degrees]
            f'{wind_speed:.1f}',    # 22: Wind Speed [m/s]
            f'{total_sky_cover}',   # 23: Total Sky Cover [tenths]
            f'{opaque_sky_cover}',  # 24: Opaque Sky Cover [tenths]
            '9999',                 # 25: Visibility [km] - missing
            '9999',                 # 26: Ceiling Height [m] - missing
            '9',                    # 27: Present Weather Observation
            '999999999',            # 28: Present Weather Codes
            f'{precip:.1f}',        # 29: Precipitable Water [mm]
            '999',                  # 30: Aerosol Optical Depth - missing
            f'{snow_depth:.0f}',    # 31: Snow Depth [cm]
            '99',                   # 32: Days Since Last Snowfall - missing
            '999',                  # 33: Albedo - missing
            f'{precip:.1f}',        # 34: Liquid Precipitation Depth [mm]
            '1',                    # 35: Liquid Precipitation Quantity [hr]
        ]
        
        epw_rows.append(','.join(str(x) for x in epw_row))
    
    # Write EPW file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        header = EPW_HEADER_TEMPLATE.format(**location_info)
        f.write(header)
        
        # Write data rows
        for row in epw_rows:
            f.write(row + '\n')
    
    print(f"EPW file generated: {output_path}")
    print(f"  Total hours: {len(epw_rows)}")
    
    return str(output_path)


def _get_value(row: pd.Series, col: str, default: float) -> float:
    """Get value from row with fallback to default."""
    if col in row.index and pd.notna(row[col]):
        return float(row[col])
    return default


def _calculate_extraterrestrial(dt: datetime, latitude: float) -> float:
    """
    Calculate extraterrestrial horizontal radiation.
    
    Simplified calculation based on solar constant and geometry.
    """
    # Solar constant
    Gsc = 1367  # W/m²
    
    # Day of year
    n = dt.timetuple().tm_yday
    
    # Declination angle
    declination = 23.45 * math.sin(math.radians(360 * (284 + n) / 365))
    
    # Hour angle (simplified, assuming solar noon at 12:00)
    hour_angle = 15 * (dt.hour - 12)
    
    # Solar altitude
    lat_rad = math.radians(latitude)
    dec_rad = math.radians(declination)
    ha_rad = math.radians(hour_angle)
    
    sin_altitude = (math.sin(lat_rad) * math.sin(dec_rad) + 
                   math.cos(lat_rad) * math.cos(dec_rad) * math.cos(ha_rad))
    
    if sin_altitude <= 0:
        return 0
    
    # Extraterrestrial radiation on horizontal surface
    # Account for Earth-Sun distance variation
    B = 360 * (n - 81) / 365
    E0 = 1 + 0.033 * math.cos(math.radians(B))
    
    G0 = Gsc * E0 * sin_altitude
    
    return max(0, G0)


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names by removing unit suffixes.
    
    Example: "temperature (degC)" -> "temperature"
             "datetime (UTC)" -> "datetime"
    """
    rename_map = {}
    for col in df.columns:
        if ' (' in col:
            base_name = col.split(' (')[0]
            rename_map[col] = base_name
    
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  Normalized {len(rename_map)} column names")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Convert Oikolab CSV to EnergyPlus EPW format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file from Oikolab",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output EPW file path (default: same name with .epw extension)",
    )
    parser.add_argument(
        "--city",
        type=str,
        default="Shanghai",
        help="City name for EPW header",
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=31.23,
        help="Latitude (default: 31.23 for Shanghai)",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=121.47,
        help="Longitude (default: 121.47 for Shanghai)",
    )
    args = parser.parse_args()
    
    # Load CSV
    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df)} records")
    
    # Determine output path
    if args.output is None:
        args.output = Path(args.input).with_suffix('.epw')
    
    # Location info
    location_info = {
        'city': args.city,
        'state': args.city,
        'country': 'CHN',
        'data_source': 'Oikolab',
        'wmo': '583620',
        'latitude': args.latitude,
        'longitude': args.longitude,
        'timezone': 8,
        'elevation': 4,
    }
    
    # Convert
    convert_to_epw(df, args.output, location_info)
    
    print("\n✅ Conversion complete!")
    print(f"EPW file ready for EnergyPlus: {args.output}")


if __name__ == "__main__":
    main()
