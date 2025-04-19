import numpy as np
import pandas as pd

from pathlib import Path

if __name__ == '__main__':
    datasets = {
        'embrace': [
            'mapas1_embrace_2022_2024_0800',
            'mapas1_embrace_2022_2024_1600',
            'mapas1_embrace_2022_2024_2000_2200_0000_0200_0400'
        ],
        'igs': [
            'mapas1_igs_2022_2024_0800',
            'mapas1_igs_2022_2024_1600',
            'mapas1_igs_2022_2024_2000_2200_0000_0200_0400',
            'mapas2_igs_2022_2024_0800',
            'mapas2_igs_2022_2024_1600',
            'mapas2_igs_2022_2024_2000_2200_0000_0200_0400'
        ],
        'maggia': [
            'mapas1_maggia_2022_2024_0800',
            'mapas1_maggia_2022_2024_1600',
            'mapas1_maggia_2022_2024_2000_2200_0000_0200_0400',
            'mapas2_maggia_2022_2024_0800',
            'mapas2_maggia_2022_2024_1600',
            'mapas2_maggia_2022_2024_2000_2200_0000_0200_0400'
        ],
        'nagoya': [
            'mapas1_nagoya_2022_2024_0800',
            'mapas1_nagoya_2022_2024_1600',
            'mapas1_nagoya_2022_2024_2000_2200_0000_0200_0400',
            'mapas2_nagoya_2022_2024_0800',
            'mapas2_nagoya_2022_2024_1600',
            'mapas2_nagoya_2022_2024_2000_2200_0000_0200_0400'
        ]

    }

    comparisons = [
        ['embrace', 'igs'],
        ['embrace', 'maggia'],
        ['embrace', 'nagoya'],
        ['igs', 'maggia'],
        ['igs', 'nagoya'],
        ['maggia', 'nagoya'],
    ]

    base_dir = Path('.').resolve() / 'output'
    dataset_type = 'interp'

    result = []
    for comparison in comparisons:
        for i, dataset_a in enumerate(datasets[comparison[0]]):
            dataset_b = datasets[comparison[1]][i]
            print(dataset_a, 'x', dataset_b)

            for file_a in sorted((base_dir / f'{dataset_a}_{dataset_type}').glob('*.npy')):
                epoch = np.datetime64(file_a.name[:-4].replace('.', ':'))
                file_b = base_dir / f'{dataset_b}_{dataset_type}' / file_a.name

                if not file_b.exists():
                    continue

                map_a = np.load(file_a).flatten()
                map_a = np.nan_to_num(map_a)

                map_b = np.load(file_b).flatten()
                map_b = np.nan_to_num(map_b)

                corr = np.corrcoef(map_a, map_b)[0, 1]

                result.append({
                    'datetime': epoch,
                    'comparison': f'{comparison[0]} x {comparison[1]}',
                    'dataset_a': dataset_a,
                    'dataset_b': dataset_b,
                    'map_a': map_a,
                    'map_b': map_b,
                    'corr': corr,
                    'corr_p': corr * 100
                })

    df = pd.DataFrame(result)
    df.to_csv('result_correlation.csv', index=False)
    df.sort_values('datetime', inplace=True)

    for comparison in comparisons:
        for i, dataset_a in enumerate(datasets[comparison[0]]):
            print(comparison[0].upper(), 'x', comparison[1].upper())

            dataset_b = datasets[comparison[1]][i]
            print(dataset_a, 'x', dataset_b)

            comparison_type = f'{comparison[0]} x {comparison[1]}'
            selection = df.loc[(df['comparison'] == comparison_type)]
            selection = selection.loc[selection['dataset_a'] == dataset_a]
            print(len(selection))

            map_a_concat = np.concatenate(selection['map_a'].values)
            map_b_concat = np.concatenate(selection['map_b'].values)
            total = np.corrcoef(map_a_concat, map_b_concat)[0, 1]


            print(f'Total correlation: {total * 100:.2f}%')

            for year in [2022, 2023, 2024]:
                mar = selection.loc[(selection['datetime'].dt.month == 3) &
                                    (selection['datetime'].dt.year == year)]
                jun = selection.loc[(selection['datetime'].dt.month == 6) &
                                    (selection['datetime'].dt.year == year)]
                sep = selection.loc[(selection['datetime'].dt.month == 9) &
                                    (selection['datetime'].dt.year == year)]
                dec = selection.loc[(selection['datetime'].dt.month == 12) &
                                    (selection['datetime'].dt.year == year)]
                if not mar.empty:
                    map_a_concat = np.concatenate(mar['map_a'].values)
                    map_b_concat = np.concatenate(mar['map_b'].values)
                    mar_total = np.corrcoef(map_a_concat, map_b_concat)[0, 1]

                    print(f'Correlation (March/{year}): {mar_total*100:.2f}%')
                if not jun.empty:
                    map_a_concat = np.concatenate(jun['map_a'].values)
                    map_b_concat = np.concatenate(jun['map_b'].values)
                    jun_total = np.corrcoef(map_a_concat, map_b_concat)[0, 1]

                    print(f'Correlation (June/{year}): {jun_total*100:.2f}%')
                if not sep.empty:
                    map_a_concat = np.concatenate(sep['map_a'].values)
                    map_b_concat = np.concatenate(sep['map_b'].values)
                    sep_total = np.corrcoef(map_a_concat, map_b_concat)[0, 1]

                    print(f'Correlation (September/{year}): {sep_total*100:.2f}%')
                if not dec.empty:
                    map_a_concat = np.concatenate(dec['map_a'].values)
                    map_b_concat = np.concatenate(dec['map_b'].values)
                    dec_total = np.corrcoef(map_a_concat, map_b_concat)[0, 1]

                    print(f'Correlation (December/{year}): {dec_total*100:.2f}%')

            print()  
    if metric_type in ['pearson', 'r2', 'cosine']:
        overall_metrics[f'avg_{metric_type}_percent'] = overall_metrics[f'avg_{metric_type}'] * 100
    
    overall_metrics.to_csv(f'dataset_ranking_{metric_type}.csv', index=False)
    print(f"\nDataset ranking saved to 'dataset_ranking_{metric_type}.csv'")
