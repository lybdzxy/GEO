import os
import xarray as xr
import traceback
from datetime import datetime

time_list = [0, 6, 12, 18]

# Define the log file
log_file = 'F:/ERA5/hourly/lvl/error_log.txt'

# Ensure the directory for the log file exists
os.makedirs(os.path.dirname(log_file), exist_ok=True)

for t in time_list:
    for year in range(1940, 2025):
        for month in range(1, 13):
            month_format = f"{month:02d}"
            time_format = f"{t:02d}"

            # Output file path
            # output_file = f'F:/ERA5/hourly/lvl/{t}z/ERA5_{t}z_lvl_{year}{month_format}.nc'
            output_file = f'F:/ERA5/hourly/lvl/50s/ERA5_{t}z_lvl_{year}{month_format}.nc'
            # Skip if output file already exists
            if os.path.exists(output_file):
                print(f"Skipping existing file: {output_file}")
                continue

            # Input file paths
            file1 = f'F:/ERA5/hourly/lvl/50s/ERA5_part_{t}z_lvl_{year}{month_format}.nc'
            file2 = f'F:/ERA5/hourly/lvl/50s/ERA5_part2_{t}z_lvl_{year}{month_format}.nc'

            # Check if input files exist
            if not os.path.exists(file1) or not os.path.exists(file2):
                print(f"Missing input files, skipping: {file1}, {file2}")
                continue

            print(f"Processing: {output_file}")

            try:
                # Load datasets
                ds1 = xr.open_dataset(file1)
                ds2 = xr.open_dataset(file2)

                # Merge datasets
                merged_ds = xr.merge([ds1, ds2])

                # Save merged dataset
                merged_ds.to_netcdf(output_file)

                # Close files
                ds1.close()
                ds2.close()
                merged_ds.close()

            except Exception as e:
                # Log error to file
                with open(log_file, 'a') as f:
                    f.write(f"[{datetime.now()}] Error processing {output_file}:\n")
                    f.write(f"Error message: {str(e)}\n")
                    f.write(f"Traceback:\n{traceback.format_exc()}\n")
                    f.write("-" * 50 + "\n")
                print(f"Error occurred for {output_file}, logged to {log_file}, continuing...")
                continue

print("All tasks completed!")