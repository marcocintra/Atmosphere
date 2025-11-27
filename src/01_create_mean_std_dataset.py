import pandas as pd

from pathlib import Path
from src.utils import decimal_hours_to_hms, create_folder


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    print(f"Project dir: {project_dir}")

    input_dir = project_dir / 'data' / 'gopi'
    print(f"Input dir: {input_dir}")

    dtype_dict = {
        'time': 'float64',
        'mean': 'float64',
        'std': 'float64',
        'lat': 'float64'
    }

    df = pd.DataFrame()
    for station_dir in sorted(input_dir.glob('Gopi_TEC_????_???_????')):
        print(station_dir)
        for data_file in sorted(station_dir.glob('*.Std')):
            print(data_file)
            station_name = data_file.name[:4]
            obs_date = data_file.with_suffix('').name[-10:]

            df_temp = pd.read_csv(data_file,
                                  delimiter='\t',
                                  header=None,
                                  na_values=['-', '- '],
                                  names=dtype_dict.keys(),
                                  dtype=dtype_dict)

            df_temp['time'] = df_temp['time'].apply(decimal_hours_to_hms, round_minutes=True).astype('str')
            df_temp['date'] = obs_date
            df_temp['datetime'] = (pd.to_datetime(df_temp['date']) + pd.to_timedelta(df_temp['time']))
            df_temp['station'] = station_name.upper()
            df = pd.concat([df, df_temp])

    df.set_index('datetime', inplace=True)

    output_dir = project_dir / 'output'
    create_folder(output_dir)
    print(f"Output dir: {output_dir}")

    filename = 'mean_std_gopi_dataset.csv'
    df.to_csv(output_dir / filename)
