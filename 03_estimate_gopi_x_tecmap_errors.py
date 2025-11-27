import numpy as np
import pandas as pd

from pathlib import Path

from src.error_estimation import GopiTecMapErrorEstimation
from src.utils import create_folder


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent
    print(f"Project dir: {project_dir}")

    amazonian_group_stations = ['AMTE', 'SAGA', 'PAAR', 'ROJI', 'RIOB', 'BELE',
                                'BOAV', 'NAUS', 'POVE']

    high_density_group_stations = ['PPTE', 'SJSP', 'CHPI', 'SPTU', 'UFPR',
                                   'SPFR', 'SPBO', 'SJRP', 'MGIN']

    igs_group_stations = ['BELE', 'BOAV', 'BRAZ', 'CUIB', 'IMPZ', 'MSGR',
                          'NAUS', 'POAL', 'POVE', 'SALU', 'SAVO', 'SPTU',
                          'TOPL', 'UFPR']

    input_dir = project_dir / 'output' / 'gopi+tecmaps'

    output_dir = project_dir / 'output' / 'errors'
    create_folder(output_dir, clear=False)

    df = pd.DataFrame()
    for file in input_dir.glob('gopi-*-data_????-??-??T??:??:??.csv'):
        df_temp = pd.read_csv(file, parse_dates=['datetime'])
        df = pd.concat([df, df_temp])

    days = sorted((df['datetime'].dt.date).unique())
    timestamps = sorted((df['datetime'].dt.time).unique())
    stations = sorted(df['station'].unique())
    networks = sorted(df['network'].unique())
    print(stations)
    print(len(stations))

    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    # TEST CASE 01 - One set of measurements per source (grouping all days and
    # GNSS stations)
    print(f'Test case 01 - {df.shape}')
    result_df = pd.DataFrame()
    for network in networks:
        print(network)
        network_df = df[df['network'] == network].copy()
        error_estimation = GopiTecMapErrorEstimation()
        if len(network_df) != 0:
            error_estimation.add_dataset(f'{network}',
                                         network_df['mean'].values,
                                         network_df['tec'].values)
        result = error_estimation.run()
        tmp_df = pd.DataFrame(result)
        tmp_df['network'] = network
        result_df = pd.concat([result_df, tmp_df])

    result_df.to_csv(output_dir / 'test_case_01-error_estimation_by_network.csv',
                     index=False)

    # TEST CASE 02 - Three sets of measurements per source (grouping all days
    # and GNSS stations), separated by <= than first quantile, >= than third
    # quantile and between first and third quantiles (interquantile), using GOPI
    # value for reference
    print(f'Test case 02 - {df.shape}')
    result_df = pd.DataFrame()
    for network in networks:
        print(network)
        network_df = df[df['network'] == network].copy()
        error_estimation = GopiTecMapErrorEstimation()
        if len(network_df) != 0:
            q1 = np.nanquantile(network_df['mean'].values, 0.25)
            q3 = np.nanquantile(network_df['mean'].values, 0.75)
            print(q1, q3)

            if len(network_df[network_df['mean'] <= q1]) != 0:
                error_estimation.add_dataset(f'{network}_q1',
                                             network_df[network_df['mean'] <= q1]['mean'].values,
                                             network_df[network_df['mean'] <= q1]['tec'].values)
            if len(network_df[network_df['mean'] >= q3]) != 0:
                error_estimation.add_dataset(f'{network}_q3',
                                             network_df[network_df['mean'] >= q3]['mean'].values,
                                             network_df[network_df['mean'] >= q3]['tec'].values)

            if len(network_df[(network_df['mean'] > q1) & (network_df['mean'] < q3)]) != 0:
                error_estimation.add_dataset(f'{network}_inter',
                                             network_df[(network_df['mean'] > q1) & (network_df['mean'] < q3)]['mean'].values,
                                             network_df[(network_df['mean'] > q1) & (network_df['mean'] < q3)]['tec'].values)

        result = error_estimation.run()
        tmp_df = pd.DataFrame(result)
        tmp_df['network'] = network
        result_df = pd.concat([result_df, tmp_df])

    result_df.to_csv(output_dir / 'test_case_02-error_estimation_by_network_and_quantile.csv',
                     index=False)

    # TEST CASE 03 - Same as test case 01, but separated by two clusters of
    # stations: Amazonian region and high density region.
    print(f'Test case 03 - {df.shape}')
    clusters = [amazonian_group_stations, high_density_group_stations]
    result_df = pd.DataFrame()
    for network in networks:
        print(network)
        network_df = df[df['network'] == network].copy()
        for cluster in clusters:
            cluster_df = network_df[network_df['station'].isin(cluster)].copy()
            error_estimation = GopiTecMapErrorEstimation()
            if len(cluster_df) != 0:
                error_estimation.add_dataset(f'{network}',
                                             cluster_df['mean'].values,
                                             cluster_df['tec'].values)
            result = error_estimation.run()
            tmp_df = pd.DataFrame(result)
            tmp_df['network'] = network
            tmp_df['cluster'] = ', '.join(cluster)
            result_df = pd.concat([result_df, tmp_df])

    result_df.to_csv(output_dir / 'test_case_03-error_estimation_by_network_and_cluster.csv',
                     index=False)

    # TEST CASE 04 - Same as test case 02, but separated by two clusters of
    # stations: Amazonian region and high density region.
    print(f'Test case 04 - {df.shape}')
    result_df = pd.DataFrame()
    for network in networks:
        print(network)
        network_df = df[df['network'] == network].copy()
        for cluster in clusters:
            cluster_df = network_df[network_df['station'].isin(cluster)].copy()
            error_estimation = GopiTecMapErrorEstimation()
            if len(cluster_df) != 0:
                q1 = np.nanquantile(cluster_df['mean'].values, 0.25)
                q3 = np.nanquantile(cluster_df['mean'].values, 0.75)
                print(q1, q3)

                if len(cluster_df[cluster_df['mean'] <= q1]) != 0:
                    error_estimation.add_dataset(f'{network}_q1',
                                                 cluster_df[cluster_df['mean'] <= q1]['mean'].values,
                                                 cluster_df[cluster_df['mean'] <= q1]['tec'].values)
                if len(cluster_df[cluster_df['mean'] >= q3]) != 0:
                    error_estimation.add_dataset(f'{network}_q3',
                                                 cluster_df[cluster_df['mean'] >= q3]['mean'].values,
                                                 cluster_df[cluster_df['mean'] >= q3]['tec'].values)

                if len(cluster_df[(cluster_df['mean'] > q1) & (cluster_df['mean'] < q3)]) != 0:
                    error_estimation.add_dataset(f'{network}_inter',
                                                 cluster_df[(cluster_df['mean'] > q1) & (cluster_df['mean'] < q3)]['mean'].values,
                                                 cluster_df[(cluster_df['mean'] > q1) & (cluster_df['mean'] < q3)]['tec'].values)

            result = error_estimation.run()
            tmp_df = pd.DataFrame(result)
            tmp_df['network'] = network
            tmp_df['cluster'] = ', '.join(cluster)
            result_df = pd.concat([result_df, tmp_df])

    result_df.to_csv(output_dir / 'test_case_04-error_estimation_by_network_cluster_and_quantile.csv',
                     index=False)

    # TEST CASE 05 - Same as test case 01, but only for IGS stations
    print(f'Test case 05 - {df.shape}')
    result_df = pd.DataFrame()
    for network in networks:
        print(network)
        network_df = df[df['network'] == network].copy()
        cluster_df = network_df[network_df['station'].isin(igs_group_stations)].copy()
        error_estimation = GopiTecMapErrorEstimation()
        if len(cluster_df) != 0:
            error_estimation.add_dataset(f'{network}',
                                         cluster_df['mean'].values,
                                         cluster_df['tec'].values)
        result = error_estimation.run()
        tmp_df = pd.DataFrame(result)
        tmp_df['network'] = network
        tmp_df['cluster'] = ', '.join(igs_group_stations)
        result_df = pd.concat([result_df, tmp_df])

    result_df.to_csv(output_dir / 'test_case_05-error_estimation_by_network_and_igs.csv',
                     index=False)

    # TEST CASE 06 - One set of measurements per source and per station
    # (grouping all GNSS stations)
    print(f'Test case 06 - {df.shape}')
    result_df = pd.DataFrame()
    for network in networks:
        print(network)
        network_df = df[df['network'] == network].copy()
        for station in stations:
            error_estimation = GopiTecMapErrorEstimation()
            station_df = network_df[network_df['station'].eq(station)].copy()

            if len(station_df) != 0:
                error_estimation.add_dataset(f'{network}_{station}',
                                             station_df['mean'].values,
                                             station_df['tec'].values)
            result = error_estimation.run()
            tmp_df = pd.DataFrame(result)
            tmp_df['network'] = network
            tmp_df['station'] = station
            result_df = pd.concat([result_df, tmp_df])

    result_df.to_csv(output_dir /
                     'test_case_06-error_estimation_by_network_and_station.csv',
                     index=False)

    # TEST CASE 07 - Same as test case 01, but only for 18 UT.
    print(f'Test case 07 - {df.shape}')

    result_df = pd.DataFrame()
    for network in networks:
        print(network)
        network_df = df[df['network'] == network].copy()
        hour_df = network_df.at_time('18:00:00').copy()
        error_estimation = GopiTecMapErrorEstimation()
        if len(hour_df) != 0:
            error_estimation.add_dataset(f'{network}',
                                         hour_df['mean'].values,
                                         hour_df['tec'].values)
        result = error_estimation.run()
        tmp_df = pd.DataFrame(result)
        tmp_df['network'] = network
        result_df = pd.concat([result_df, tmp_df])

    result_df.to_csv(output_dir / 'test_case_07-error_estimation_by_network_at_18ut.csv',
                     index=False)

    # TEST CASE 08 - TSame as test case 02, but only for 18 UT.
    print(f'Test case 08 - {df.shape}')
    result_df = pd.DataFrame()
    for network in networks:
        print(network)
        network_df = df[df['network'] == network].copy()
        hour_df = network_df.at_time('18:00:00').copy()
        error_estimation = GopiTecMapErrorEstimation()
        if len(hour_df) != 0:
            q1 = np.nanquantile(hour_df['mean'].values, 0.25)
            q3 = np.nanquantile(hour_df['mean'].values, 0.75)
            print(q1, q3)

            if len(hour_df[hour_df['mean'] <= q1]) != 0:
                error_estimation.add_dataset(f'{network}_q1',
                                             hour_df[hour_df['mean'] <= q1]['mean'].values,
                                             hour_df[hour_df['mean'] <= q1]['tec'].values)
            if len(hour_df[hour_df['mean'] >= q3]) != 0:
                error_estimation.add_dataset(f'{network}_q3',
                                             hour_df[hour_df['mean'] >= q3]['mean'].values,
                                             hour_df[hour_df['mean'] >= q3]['tec'].values)

            if len(hour_df[(hour_df['mean'] > q1) & (hour_df['mean'] < q3)]) != 0:
                error_estimation.add_dataset(f'{network}_inter',
                                             hour_df[(hour_df['mean'] > q1) & (hour_df['mean'] < q3)]['mean'].values,
                                             hour_df[(hour_df['mean'] > q1) & (hour_df['mean'] < q3)]['tec'].values)

        result = error_estimation.run()
        tmp_df = pd.DataFrame(result)
        tmp_df['network'] = network
        result_df = pd.concat([result_df, tmp_df])

    result_df.to_csv(output_dir / 'test_case_08-error_estimation_by_network_and_quantile_at_18ut.csv',
                     index=False)
