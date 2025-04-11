import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
import warnings

# Suprimir avisos para melhor legibilidade
warnings.filterwarnings('ignore')

# Função para calcular o RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Função para calcular o RMSE como porcentagem da amplitude
def calculate_rmse_percent(y_true, y_pred):
    rmse = calculate_rmse(y_true, y_pred)
    # Combinamos ambos os arrays para encontrar a amplitude total
    combined = np.concatenate([y_true, y_pred])
    data_range = np.max(combined) - np.min(combined)
    # Evita divisão por zero
    if data_range == 0:
        return 0
    return (rmse / data_range) * 100

# Funções para a transformação Z de Fisher
def fisher_z_transform(r):
    """Transforma correlação r para valor z"""
    # Limita r para evitar problemas com valores extremos (-1 ou 1)
    if r >= 1.0:
        r = 0.9999
    elif r <= -1.0:
        r = -0.9999
    return 0.5 * np.log((1 + r) / (1 - r))

def fisher_z_inverse(z):
    """Transforma z de volta para correlação r"""
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

# Funções para as métricas
def calculate_residual_error(y_true, y_pred):
    """Calcula o erro residual médio"""
    return np.mean(y_true - y_pred)

def calculate_max_residual_error(y_true, y_pred):
    """Calcula o erro residual máximo (em valor absoluto)"""
    return np.max(np.abs(y_true - y_pred))

def calculate_min_residual_error(y_true, y_pred):
    """Calcula o erro residual no 5º percentil (em valor absoluto)"""
    return np.percentile(np.abs(y_true - y_pred), 5)

def calculate_r2_score(y_true, y_pred):
    """Calcula o coeficiente de determinação (R²)"""
    return r2_score(y_true, y_pred)

def calculate_mse(y_true, y_pred):
    """Calcula o Mean Squared Error"""
    return mean_squared_error(y_true, y_pred)

def calculate_mae(y_true, y_pred):
    """Calcula o Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)

def calculate_cosine_similarity(y_true, y_pred):
    """Calcula a similaridade de cosseno entre dois vetores"""
    dot_product = np.dot(y_true, y_pred)
    norm_y_true = np.linalg.norm(y_true)
    norm_y_pred = np.linalg.norm(y_pred)
    # Evita divisão por zero
    if norm_y_true == 0 or norm_y_pred == 0:
        return 0
    return dot_product / (norm_y_true * norm_y_pred)

def calculate_huber_loss(y_true, y_pred, delta=1.0):
    """Calcula a perda de Huber com delta padrão = 1.0"""
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    quadratic = np.minimum(abs_errors, delta)
    linear = abs_errors - quadratic
    return np.mean(0.5 * quadratic * quadratic + delta * linear)

def calculate_ssim(y_true, y_pred):
    """
    Calcula o Structural Similarity Index (SSIM) entre duas imagens.
    """
    # Converte para escala de cinza se a imagem for colorida
    if len(y_true.shape) > 2 and y_true.shape[2] > 1:
        # Média dos canais para obter escala de cinza
        y_true = np.mean(y_true, axis=2)
    
    if len(y_pred.shape) > 2 and y_pred.shape[2] > 1:
        y_pred = np.mean(y_pred, axis=2)
        
    # Normaliza os dados para o intervalo [0, 1] para melhor funcionamento do SSIM
    data_range = max(np.max(y_true) - np.min(y_true), np.max(y_pred) - np.min(y_pred))
    if data_range == 0:
        data_range = 1  # Evita divisão por zero
        
    try:
        return ssim(y_true, y_pred, data_range=data_range)
    except Exception as e:
        print(f"Erro ao calcular SSIM: {str(e)}")
        print(f"Shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
        # Tenta redimensionar se os tamanhos forem diferentes
        if y_true.shape != y_pred.shape:
            print("Tentando redimensionar imagens para o mesmo tamanho...")
            min_height = min(y_true.shape[0], y_pred.shape[0])
            min_width = min(y_true.shape[1], y_pred.shape[1])
            y_true_resized = y_true[:min_height, :min_width]
            y_pred_resized = y_pred[:min_height, :min_width]
            return ssim(y_true_resized, y_pred_resized, data_range=data_range)
        return 0

def load_image(filepath):
    """Carrega uma imagem e a converte para array numpy."""
    if filepath.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        try:
            img = imread(filepath)
            # Se a imagem é colorida (RGB), converte para escala de cinza
            if len(img.shape) > 2 and img.shape[2] > 1:
                img = np.mean(img, axis=2)
            return img
        except Exception as e:
            print(f"Erro ao carregar imagem {filepath}: {str(e)}")
            return None
    elif filepath.suffix.lower() == '.npy':
        try:
            return np.load(filepath)
        except Exception as e:
            print(f"Erro ao carregar arquivo npy {filepath}: {str(e)}")
            return None
    else:
        print(f"Formato de arquivo não suportado: {filepath}")
        return None

# Função para calcular estatísticas de forma garantidamente consistente
def calculate_strict_stats(map_a, map_b):
    """
    Calcula rigorosamente estatísticas para dois arrays, garantindo que as estatísticas
    combinadas sejam matematicamente consistentes.
    
    Args:
        map_a: Primeiro array de dados
        map_b: Segundo array de dados
        
    Returns:
        Dict com todas as estatísticas
    """
    # Aplainar arrays se necessário
    if len(map_a.shape) > 1:
        map_a_flat = map_a.flatten()
    else:
        map_a_flat = map_a
        
    if len(map_b.shape) > 1:
        map_b_flat = map_b.flatten()
    else:
        map_b_flat = map_b
    
    # Calcular estatísticas individuais
    min_a = float(np.min(map_a_flat))
    max_a = float(np.max(map_a_flat))
    q3_a = float(np.percentile(map_a_flat, 75))
    
    min_b = float(np.min(map_b_flat))
    max_b = float(np.max(map_b_flat))
    q3_b = float(np.percentile(map_b_flat, 75))
    
    # Calcular estatísticas combinadas GARANTIDAMENTE consistentes
    min_both = min(min_a, min_b)
    max_both = max(max_a, max_b)
    
    # Criar array combinado para calcular Q3
    both_flats = np.concatenate([map_a_flat, map_b_flat])
    q3_both = float(np.percentile(both_flats, 75))
    
    # Amplitude de dados
    data_range = max_both - min_both
    
    return {
        'min_a': min_a,
        'max_a': max_a,
        'q3_a': q3_a,
        'min_b': min_b,
        'max_b': max_b,
        'q3_b': q3_b,
        'min_both': min_both,
        'max_both': max_both,
        'q3_both': q3_both,
        'data_range': data_range
    }

if __name__ == '__main__':
    # Configuração do parser de argumentos
    parser = argparse.ArgumentParser(description='Calculate metrics between datasets')
    parser.add_argument('--metric', type=str, 
                        choices=['pearson', 'rmse', 'residual', 'max_residual', 'min_residual', 
                                'r2', 'mse', 'mae', 'cosine', 'huber', 'ssim'], 
                        default='pearson',
                        help='Metric to calculate: pearson, rmse, residual, max_residual, min_residual, r2, mse, mae, cosine, huber, or ssim')
    parser.add_argument('--huber-delta', type=float, default=1.0,
                        help='Delta parameter for Huber loss (default: 1.0)')
    parser.add_argument('--min-residual-percentile', type=float, default=5.0,
                        help='Percentile for min_residual calculation (default: 5.0)')
    parser.add_argument('--dataset-suffix', type=str, default=None,
                        help='Override the dataset suffix (default: determined by metric)')
    parser.add_argument('--verify-stats', action='store_true',
                        help='Verify strict consistency of statistics')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top maps to display (default: 10)')
    args = parser.parse_args()
    
    metric_type = args.metric
    huber_delta = args.huber_delta
    min_residual_percentile = args.min_residual_percentile
    verify_stats = args.verify_stats
    top_n = args.top_n
    
    # Determinar o tipo de dataset correto baseado na métrica
    if args.dataset_suffix:
        # Se especificado manualmente, use esse
        dataset_suffix = args.dataset_suffix
    else:
        # Por padrão, use 'interp' para todas as métricas exceto SSIM
        dataset_suffix = 'interp_raster' if metric_type == 'ssim' else 'interp'
    
    # Determinar se a métrica é do tipo "quanto maior, melhor" ou "quanto menor, melhor"
    higher_is_better = metric_type in ['pearson', 'r2', 'cosine', 'ssim']
    
    # Determinar se a métrica precisa de transformação Fisher para média
    needs_fisher_transform = metric_type in ['pearson', 'r2']
    
    # Lista base de datasets
    base_datasets = {
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
    
    # Adicionar sufixo aos nomes dos datasets
    datasets = {}
    for source, dataset_list in base_datasets.items():
        datasets[source] = [f"{dataset}_{dataset_suffix}" for dataset in dataset_list]

    comparisons = [
        ['embrace', 'igs'],
        ['embrace', 'maggia'],
        ['embrace', 'nagoya'],
        ['igs', 'maggia'],
        ['igs', 'nagoya'],
        ['maggia', 'nagoya'],
    ]

    base_dir = Path('.').resolve() / 'output'
    if not base_dir.exists():
        base_dir = Path('.').resolve()  # Tenta o diretório atual se 'output' não existir

    print(f"Calculating {metric_type.upper()} metrics with additional statistics...")
    print(f"Using dataset suffix: '{dataset_suffix}' based on metric type: {metric_type}")
    print(f"Higher values are better: {'YES' if higher_is_better else 'NO'}")
    print(f"Using Fisher Z transform: {'YES' if needs_fisher_transform else 'NO'}")
    print(f"Using strict statistics calculation: {'YES' if verify_stats else 'NO'}")
    print(f"Displaying top {top_n} maps for each comparison")
    
    # Verificar quais diretórios existem
    existing_dirs = []
    for path in base_dir.glob('*'):
        if path.is_dir() and dataset_suffix in path.name:
            existing_dirs.append(path)
    
    if not existing_dirs:
        print(f"WARNING: No directories found with suffix '{dataset_suffix}' in {base_dir}")
        print("Available directories:")
        for path in base_dir.glob('*'):
            if path.is_dir():
                print(f"  - {path.name}")
    else:
        print(f"Found {len(existing_dirs)} directories with the expected suffix.")
        for path in existing_dirs[:5]:  # Mostra apenas os primeiros 5 para não sobrecarregar a saída
            print(f"  - {path.name}")
        if len(existing_dirs) > 5:
            print(f"  ... and {len(existing_dirs) - 5} more.")

    # Contador para arquivos processados
    processed_files = 0
    result = []

    for comparison in comparisons:
        for i in range(min(len(datasets[comparison[0]]), len(datasets[comparison[1]]))):
            dataset_a = datasets[comparison[0]][i]
            dataset_b = datasets[comparison[1]][i]
            
            print(f"\nProcessing {dataset_a} x {dataset_b}")
            
            # Verifique se os diretórios existem
            dir_a = base_dir / dataset_a
            dir_b = base_dir / dataset_b
            
            if not dir_a.exists():
                print(f"WARNING: Directory not found: {dir_a}")
                continue
            
            if not dir_b.exists():
                print(f"WARNING: Directory not found: {dir_b}")
                continue

            # Processamento específico para SSIM (arquivos de imagem)
            if metric_type == 'ssim':
                # Lista de arquivos de imagens no diretório A
                files_a = []
                for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                    files_a.extend(list(dir_a.glob(f'*{ext}')))
                
                files_a = sorted(files_a)
                
                if not files_a:
                    print(f"WARNING: No image files found in {dir_a}")
                    continue
                    
                print(f"Found {len(files_a)} image files in {dataset_a}")
                
                # Para cada arquivo no dataset A, encontre o arquivo correspondente no dataset B
                for file_a in files_a:
                    # Extrair o timestamp do nome do arquivo
                    try:
                        # Tenta extrair a parte da data/hora do nome do arquivo
                        # Assumindo formato como "2022-03-01T08.00.00_raster.png"
                        timestamp = file_a.stem.split('_')[0]
                        
                        # Procura um arquivo no dataset B com o mesmo timestamp
                        matching_files_b = list(dir_b.glob(f"{timestamp}*"))
                        
                        if not matching_files_b:
                            continue
                            
                        file_b = matching_files_b[0]
                        
                        # Carregar as imagens
                        img_a = load_image(file_a)
                        img_b = load_image(file_b)
                        
                        if img_a is None or img_b is None:
                            continue
                        
                        processed_files += 1
                        
                        # Calcular SSIM
                        ssim_value = calculate_ssim(img_a, img_b)
                        
                        # Calcular estatísticas de forma estritamente consistente
                        stats = calculate_strict_stats(img_a, img_b)
                        
                        # Verificar consistência se solicitado
                        if verify_stats:
                            # Calcular mínimo e máximo combinados de forma tradicional
                            both_imgs = np.concatenate([img_a.flatten(), img_b.flatten()])
                            trad_min = np.min(both_imgs)
                            trad_max = np.max(both_imgs)
                            
                            # Verificar consistência
                            if abs(stats['min_both'] - trad_min) > 1e-10 or abs(stats['max_both'] - trad_max) > 1e-10:
                                print(f"INCONSISTENCY DETECTED in file {file_a.name}:")
                                print(f"  Min A: {stats['min_a']:.6f}, Min B: {stats['min_b']:.6f}")
                                print(f"  Strict Min Both: {stats['min_both']:.6f}, Traditional Min Both: {trad_min:.6f}")
                                print(f"  Max A: {stats['max_a']:.6f}, Max B: {stats['max_b']:.6f}")
                                print(f"  Strict Max Both: {stats['max_both']:.6f}, Traditional Max Both: {trad_max:.6f}")
                        
                        # Criar dados de resultado
                        try:
                            epoch = np.datetime64(timestamp.replace('T', ' ').replace('.', ':'))
                        except:
                            epoch = np.datetime64('1970-01-01T00:00:00')
                            print(f"Warning: Could not parse datetime from filename {file_a.name}")
                        
                        result_data = {
                            'datetime': epoch,
                            'comparison': f'{comparison[0]} x {comparison[1]}',
                            'dataset_a': dataset_a,
                            'dataset_b': dataset_b,
                            'source_a': comparison[0],
                            'source_b': comparison[1],
                            'filename_a': file_a.name,
                            'filename_b': file_b.name,
                            metric_type: ssim_value,
                            f'{metric_type}_p': ssim_value * 100,  # percentual para SSIM
                            # Estatísticas estritamente consistentes
                            'min_a': stats['min_a'],
                            'max_a': stats['max_a'],
                            'q3_a': stats['q3_a'],
                            'min_b': stats['min_b'],
                            'max_b': stats['max_b'],
                            'q3_b': stats['q3_b'],
                            'min_both': stats['min_both'],
                            'max_both': stats['max_both'],
                            'q3_both': stats['q3_both'],
                            'data_range': stats['data_range']
                        }
                        
                        result.append(result_data)
                        
                        # Mostrar progresso a cada 10 arquivos
                        if processed_files % 10 == 0:
                            print(f"Processed {processed_files} file pairs...")
                    
                    except Exception as e:
                        print(f"Error processing files {file_a}: {str(e)}")
            
            # Processamento para outras métricas (arquivos .npy)
            else:
                # Lista de arquivos binários .npy no diretório A
                files_a = sorted(list(dir_a.glob('*.npy')))
                
                if not files_a:
                    print(f"WARNING: No .npy files found in {dir_a}")
                    continue
                    
                print(f"Found {len(files_a)} .npy files in {dataset_a}")
                
                # Para cada arquivo no dataset A, encontre o arquivo correspondente no dataset B
                for file_a in files_a:
                    file_b = dir_b / file_a.name
                    
                    if not file_b.exists():
                        continue
                    
                    try:
                        # Carregar os dados
                        map_a = np.load(file_a)
                        map_b = np.load(file_b)
                        
                        # Garantir que não há NaNs
                        map_a = np.nan_to_num(map_a)
                        map_b = np.nan_to_num(map_b)
                        
                        # Aplainar os arrays para cálculo das métricas
                        map_a_flat = map_a.flatten()
                        map_b_flat = map_b.flatten()
                        
                        processed_files += 1
                        
                        # Obter a função de métrica apropriada
                        metric_function = None
                        if metric_type == 'pearson':
                            metric_value = np.corrcoef(map_a_flat, map_b_flat)[0, 1]
                        elif metric_type == 'rmse':
                            metric_value = calculate_rmse(map_a_flat, map_b_flat)
                        elif metric_type == 'residual':
                            metric_value = calculate_residual_error(map_a_flat, map_b_flat)
                        elif metric_type == 'max_residual':
                            metric_value = calculate_max_residual_error(map_a_flat, map_b_flat)
                        elif metric_type == 'min_residual':
                            metric_value = calculate_min_residual_error(map_a_flat, map_b_flat)
                        elif metric_type == 'r2':
                            metric_value = calculate_r2_score(map_a_flat, map_b_flat)
                        elif metric_type == 'mse':
                            metric_value = calculate_mse(map_a_flat, map_b_flat)
                        elif metric_type == 'mae':
                            metric_value = calculate_mae(map_a_flat, map_b_flat)
                        elif metric_type == 'cosine':
                            metric_value = calculate_cosine_similarity(map_a_flat, map_b_flat)
                        elif metric_type == 'huber':
                            metric_value = calculate_huber_loss(map_a_flat, map_b_flat, huber_delta)
                        
                        # Calcular estatísticas de forma estritamente consistente
                        stats = calculate_strict_stats(map_a, map_b)
                        
                        # Verificar consistência se solicitado
                        if verify_stats:
                            # Calcular mínimo e máximo combinados de forma tradicional
                            both_maps = np.concatenate([map_a_flat, map_b_flat])
                            trad_min = np.min(both_maps)
                            trad_max = np.max(both_maps)
                            
                            # Verificar consistência
                            if abs(stats['min_both'] - trad_min) > 1e-10 or abs(stats['max_both'] - trad_max) > 1e-10:
                                print(f"INCONSISTENCY DETECTED in file {file_a.name}:")
                                print(f"  Min A: {stats['min_a']:.6f}, Min B: {stats['min_b']:.6f}")
                                print(f"  Strict Min Both: {stats['min_both']:.6f}, Traditional Min Both: {trad_min:.6f}")
                                print(f"  Max A: {stats['max_a']:.6f}, Max B: {stats['max_b']:.6f}")
                                print(f"  Strict Max Both: {stats['max_both']:.6f}, Traditional Max Both: {trad_max:.6f}")
                        
                        # Extrair data e hora do nome do arquivo
                        try:
                            epoch = np.datetime64(file_a.stem.replace('.', ':'))
                        except:
                            # Se falhar, usar um valor padrão para ordenação
                            epoch = np.datetime64('1970-01-01T00:00:00')
                            print(f"Warning: Could not parse datetime from filename {file_a.name}")
                        
                        # Calcular valor percentual para métricas normalizadas
                        if metric_type in ['pearson', 'r2', 'cosine']:
                            value_p = metric_value * 100  # Valor percentual
                        elif metric_type == 'rmse' and stats['data_range'] != 0:
                            value_p = (metric_value / stats['data_range']) * 100  # RMSE como % do range
                        else:
                            value_p = metric_value  # Usar valor bruto para outras métricas
                        
                        result_data = {
                            'datetime': epoch,
                            'comparison': f'{comparison[0]} x {comparison[1]}',
                            'dataset_a': dataset_a,
                            'dataset_b': dataset_b,
                            'source_a': comparison[0],
                            'source_b': comparison[1],
                            'filename': file_a.name,
                            metric_type: metric_value,
                            f'{metric_type}_p': value_p,
                            # Estatísticas estritamente consistentes
                            'min_a': stats['min_a'],
                            'max_a': stats['max_a'],
                            'q3_a': stats['q3_a'],
                            'min_b': stats['min_b'],
                            'max_b': stats['max_b'],
                            'q3_b': stats['q3_b'],
                            'min_both': stats['min_both'],
                            'max_both': stats['max_both'],
                            'q3_both': stats['q3_both'],
                            'data_range': stats['data_range']
                        }
                        
                        # Para análises posteriores
                        result_data['map_a'] = map_a_flat
                        result_data['map_b'] = map_b_flat
                        
                        result.append(result_data)
                        
                        # Mostrar progresso a cada 10 arquivos
                        if processed_files % 10 == 0:
                            print(f"Processed {processed_files} file pairs...")
                    
                    except Exception as e:
                        print(f"Error processing files {file_a} and {file_b}: {str(e)}")
                        import traceback
                        traceback.print_exc()

    # Verificar se temos resultados para processar
    if not result:
        print("\nERROR: No data found for analysis. Please check if your files exist.")
        print(f"Make sure you have directories with suffix '{dataset_suffix}' containing appropriate files.")
        print("For SSIM, you need .png/.jpg files in 'interp_raster' directories.")
        print("For other metrics, you need .npy files in 'interp' directories.")
        exit(1)

    print(f"\nProcessed {processed_files} file pairs successfully.")

    # Criar o DataFrame
    df = pd.DataFrame(result)

    # Salvar os resultados brutos
    df.to_csv(f'result_{metric_type}_with_stats.csv', index=False)
    
    # Ordenar por data
    if 'datetime' in df.columns:
        df.sort_values('datetime', inplace=True)

    # Dictionary to store aggregate metrics for each dataset
    dataset_metrics = defaultdict(list)
    dataset_total_metrics = defaultdict(float)
    dataset_count = defaultdict(int)
    
    # Dicionário para armazenar top N mapas
    top_maps_by_comparison = {}

    # Analisar resultados por comparação
    for comparison in comparisons:
        for i in range(min(len(datasets[comparison[0]]), len(datasets[comparison[1]]))):
            dataset_a = datasets[comparison[0]][i]
            dataset_b = datasets[comparison[1]][i]
            
            comparison_type = f'{comparison[0]} x {comparison[1]}'
            selection = df.loc[(df['comparison'] == comparison_type)]
            selection = selection.loc[(selection['dataset_a'] == dataset_a) & (selection['dataset_b'] == dataset_b)]
            
            if selection.empty:
                print(f"No data found for {dataset_a} x {dataset_b}")
                continue
                
            print(f"\n{comparison[0].upper()} x {comparison[1].upper()}")
            print(f"{dataset_a} x {dataset_b}")
            print(f"Number of comparisons: {len(selection)}")
            
            # Calcular métrica média para esta comparação
            metric_values = selection[metric_type].values
            metric_value = np.mean(metric_values)
            
            # Armazenar valores para ranking
            dataset_metrics[comparison[0]].extend(metric_values)
            dataset_metrics[comparison[1]].extend(metric_values)
            
            dataset_total_metrics[comparison[0]] += sum(metric_values)
            dataset_total_metrics[comparison[1]] += sum(metric_values)
            dataset_count[comparison[0]] += len(metric_values)
            dataset_count[comparison[1]] += len(metric_values)
            
            # Calcular estatísticas médias garantidamente consistentes
            mean_min_a = selection['min_a'].mean()
            mean_max_a = selection['max_a'].mean()
            mean_q3_a = selection['q3_a'].mean()
            
            mean_min_b = selection['min_b'].mean()
            mean_max_b = selection['max_b'].mean()
            mean_q3_b = selection['q3_b'].mean()
            
            # Garantir consistência nas médias combinadas
            mean_min_both = min(mean_min_a, mean_min_b)
            mean_max_both = max(mean_max_a, mean_max_b)
            mean_q3_both = selection['q3_both'].mean()
            mean_data_range = mean_max_both - mean_min_both
            
            # Mostrar estatísticas agregadas
            print("\nStatistics for All Data:")
            print(f"Dataset A - Min: {mean_min_a:.4f}, Max: {mean_max_a:.4f}, Q3: {mean_q3_a:.4f}")
            print(f"Dataset B - Min: {mean_min_b:.4f}, Max: {mean_max_b:.4f}, Q3: {mean_q3_b:.4f}")
            print(f"Combined  - Min: {mean_min_both:.4f}, Max: {mean_max_both:.4f}, Q3: {mean_q3_both:.4f}")
            print(f"Data Range: {mean_data_range:.4f}")
            
            # Exibir a métrica com formatação adequada
            if metric_type == 'ssim':
                print(f'Average Structural Similarity Index: {metric_value:.4f} ({metric_value * 100:.2f}%)')
            elif metric_type == 'pearson':
                print(f'Average Pearson Correlation: {metric_value:.4f} ({metric_value * 100:.2f}%)')
            elif metric_type == 'r2':
                print(f'Average R² Score: {metric_value:.4f} ({metric_value * 100:.2f}%)')
            elif metric_type == 'cosine':
                print(f'Average Cosine Similarity: {metric_value:.4f} ({metric_value * 100:.2f}%)')
            elif metric_type == 'rmse':
                rmse_percent = (metric_value / mean_data_range) * 100 if mean_data_range != 0 else 0
                print(f'Average RMSE: {metric_value:.4f} ({rmse_percent:.2f}% of data range)')
            elif metric_type == 'residual':
                print(f'Average Residual Error: {metric_value:.4f}')
            elif metric_type == 'max_residual':
                print(f'Average Maximum Residual Error: {metric_value:.4f}')
            elif metric_type == 'min_residual':
                print(f'Average Minimum Residual Error ({min_residual_percentile}th percentile): {metric_value:.4f}')
            elif metric_type == 'mse':
                print(f'Average Mean Squared Error: {metric_value:.4f}')
            elif metric_type == 'mae':
                print(f'Average Mean Absolute Error: {metric_value:.4f}')
            elif metric_type == 'huber':
                print(f'Average Huber Loss (delta={huber_delta}): {metric_value:.4f}')
            
            # Extrair os Top N mapas para esta comparação
            if higher_is_better:
                # Ordenar do maior para o menor (melhor para pior)
                sorted_maps = selection.sort_values(by=metric_type, ascending=False).head(top_n)
            else:
                # Ordenar do menor para o maior (melhor para pior)
                sorted_maps = selection.sort_values(by=metric_type, ascending=True).head(top_n)
            
            # Exibir os Top N mapas
            comp_key = f"{dataset_a} x {dataset_b}"
            top_maps_by_comparison[comp_key] = sorted_maps
            
            # Exibir lista dos melhores mapas
            print(f"\nTop {top_n} Maps with Best {metric_type.upper()} Values:")
            print("-" * 80)
            for idx, row in enumerate(sorted_maps.itertuples(), 1):
                # Obter o nome do arquivo correto dependendo do tipo de métrica
                if metric_type == 'ssim':
                    file_a = row.filename_a
                    file_b = row.filename_b
                    file_info = f"{file_a} & {file_b}"
                else:
                    file_info = getattr(row, 'filename', 'Unknown')
                
                # Formatar o valor da métrica adequadamente
                if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
                    metric_display = f"{getattr(row, metric_type):.4f} ({getattr(row, f'{metric_type}_p'):.2f}%)"
                else:
                    metric_display = f"{getattr(row, metric_type):.4f}"
                
                # Verificar se há uma coluna datetime
                if hasattr(row, 'datetime'):
                    # Converter para string com formato legível
                    try:
                        date_str = pd.to_datetime(row.datetime).strftime('%Y-%m-%d %H:%M')
                    except:
                        date_str = str(row.datetime)
                    print(f"{idx}. {date_str} - {file_info}: {metric_display}")
                else:
                    print(f"{idx}. {file_info}: {metric_display}")
            print("-" * 80)
            
            # Análise mensal para dados com datetime
            if 'datetime' in selection.columns:
                for year in [2022, 2023, 2024]:
                    for month in range(1, 13):
                        month_data = selection.loc[(selection['datetime'].dt.month == month) &
                                               (selection['datetime'].dt.year == year)]
                        
                        if not month_data.empty:
                            month_name = pd.Timestamp(year=year, month=month, day=1).strftime('%B/%Y')
                            month_metric = month_data[metric_type].mean()
                            
                            if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
                                print(f'{month_name}: {metric_type.upper()} = {month_metric:.4f} ({month_metric * 100:.2f}%)')
                            else:
                                print(f'{month_name}: {metric_type.upper()} = {month_metric:.4f}')

    # Calcular métricas médias para cada dataset
    dataset_avg_metrics = {}
    
    # Usar transformação de Fisher se necessário
    if needs_fisher_transform:
        # Criar valores Z para cada dataset
        dataset_z_values = defaultdict(list)
        
        for dataset, values in dataset_metrics.items():
            for val in values:
                try:
                    z_value = fisher_z_transform(val)
                    dataset_z_values[dataset].append(z_value)
                except:
                    pass
                    
        # Calcular média dos valores Z e transformar de volta
        for dataset in dataset_z_values:
            if dataset_z_values[dataset]:
                avg_z = np.mean(dataset_z_values[dataset])
                dataset_avg_metrics[dataset] = fisher_z_inverse(avg_z)
            else:
                dataset_avg_metrics[dataset] = 0
        
        print("\nUsing Fisher Z transformation for averaging.")
    else:
        # Médias simples para outras métricas
        for dataset in dataset_total_metrics:
            if dataset_count[dataset] > 0:
                dataset_avg_metrics[dataset] = dataset_total_metrics[dataset] / dataset_count[dataset]
            else:
                dataset_avg_metrics[dataset] = 0
    
    # Encontrar o melhor dataset
    if higher_is_better:
        best_dataset = max(dataset_avg_metrics.items(), key=lambda x: x[1])
        worst_dataset = min(dataset_avg_metrics.items(), key=lambda x: x[1])
    else:
        best_dataset = min(dataset_avg_metrics.items(), key=lambda x: x[1])
        worst_dataset = max(dataset_avg_metrics.items(), key=lambda x: x[1])
    
    # Determinar nome apropriado para a métrica
    if metric_type == 'ssim':
        metric_name = "STRUCTURAL SIMILARITY INDEX"
    elif metric_type == 'pearson':
        metric_name = "PEARSON CORRELATION"
    elif metric_type == 'r2':
        metric_name = "R² SCORE"
    elif metric_type == 'rmse':
        metric_name = "ROOT MEAN SQUARED ERROR"
    elif metric_type == 'residual':
        metric_name = "RESIDUAL ERROR"
    elif metric_type == 'max_residual':
        metric_name = "MAXIMUM RESIDUAL ERROR"
    elif metric_type == 'min_residual':
        metric_name = f"MINIMUM RESIDUAL ERROR ({min_residual_percentile}th PERCENTILE)"
    elif metric_type == 'mse':
        metric_name = "MEAN SQUARED ERROR"
    elif metric_type == 'mae':
        metric_name = "MEAN ABSOLUTE ERROR"
    elif metric_type == 'cosine':
        metric_name = "COSINE SIMILARITY"
    elif metric_type == 'huber':
        metric_name = f"HUBER LOSS (delta={huber_delta})"
    else:
        metric_name = metric_type.upper()
    
    print(f"\n===== DATASET {metric_name} ANALYSIS =====")
    
    if higher_is_better:
        print(f"Best dataset: {best_dataset[0]} with average {metric_type} of {best_dataset[1]:.4f}")
        print(f"Worst dataset: {worst_dataset[0]} with average {metric_type} of {worst_dataset[1]:.4f}")
        
        # Adicionar porcentagem para métricas específicas
        if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
            print(f"Best percentage: {best_dataset[1] * 100:.2f}%")
            print(f"Worst percentage: {worst_dataset[1] * 100:.2f}%")
    else:
        print(f"Best dataset (lowest error): {best_dataset[0]} with average {metric_type} of {best_dataset[1]:.4f}")
        print(f"Worst dataset (highest error): {worst_dataset[0]} with average {metric_type} of {worst_dataset[1]:.4f}")
    
    # Imprimir ranking de todos os datasets
    print("\nDataset Ranking (from best to worst):")
    if higher_is_better:
        sorted_datasets = sorted(dataset_avg_metrics.items(), key=lambda x: x[1], reverse=True)
    else:
        sorted_datasets = sorted(dataset_avg_metrics.items(), key=lambda x: x[1])
        
    for i, (dataset, avg_val) in enumerate(sorted_datasets, 1):
        if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
            print(f"{i}. {dataset}: {avg_val:.4f} ({avg_val * 100:.2f}%)")
        else:
            print(f"{i}. {dataset}: {avg_val:.4f}")
    
    # Encontrar os Top N mapas globais (entre todos os conjuntos de comparações)
    print(f"\n===== TOP {top_n} MAPS OVERALL =====")
    if higher_is_better:
        top_overall = df.sort_values(by=metric_type, ascending=False).head(top_n)
    else:
        top_overall = df.sort_values(by=metric_type, ascending=True).head(top_n)

    print("-" * 100)

    for idx, row in enumerate(top_overall.itertuples(), 1):
        # Obter informações detalhadas dos datasets
        dataset_a = row.dataset_a
        dataset_b = row.dataset_b
    
        # Obter informações do arquivo corretas
        if metric_type == 'ssim':
            file_a = row.filename_a
            file_b = row.filename_b
            file_info = f"{file_a} & {file_b}"
        else:
            file_info = getattr(row, 'filename', 'Unknown')
    
        # Formatar o valor da métrica adequadamente
        if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
            metric_display = f"{getattr(row, metric_type):.4f} ({getattr(row, f'{metric_type}_p'):.2f}%)"
        else:
            metric_display = f"{getattr(row, metric_type):.4f}"
    
        # Verificar se há uma coluna datetime
        if hasattr(row, 'datetime'):
            # Converter para string com formato legível
            try:
                date_str = pd.to_datetime(row.datetime).strftime('%Y-%m-%d %H:%M')
            except:
                date_str = str(row.datetime)
        
            # Mostrar informações completas
            print(f"{idx}. Data: {date_str}")
            print(f"   Comparação: {dataset_a} x {dataset_b}")
            print(f"   Arquivos: {file_info}")
            print(f"   {metric_type.upper()}: {metric_display}")
            if idx < len(top_overall):
                print("-" * 50)  # Separador entre entradas
        else:
            # Versão sem datetime
            print(f"{idx}. Comparação: {dataset_a} x {dataset_b}")
            print(f"   Arquivos: {file_info}")
            print(f"   {metric_type.upper()}: {metric_display}")
            if idx < len(top_overall):
                print("-" * 50)  # Separador entre entradas
            print("-" * 100)
    
    # Salvar métricas gerais em um arquivo separado
    overall_metrics = pd.DataFrame({
        'dataset': list(dataset_avg_metrics.keys()),
        f'avg_{metric_type}': list(dataset_avg_metrics.values())
    })
    
    if metric_type in ['pearson', 'r2', 'cosine', 'ssim']:
        overall_metrics[f'avg_{metric_type}_percent'] = [v * 100 for v in dataset_avg_metrics.values()]
    
    overall_metrics.to_csv(f'dataset_ranking_{metric_type}.csv', index=False)
    print(f"\nDataset ranking saved to 'dataset_ranking_{metric_type}.csv'")
    
    # Resumo final
    print("\n===== COMPUTATION SUMMARY =====")
    print(f"Metric: {metric_name}")
    print(f"Dataset type: {dataset_suffix}")
    print(f"Files processed: {processed_files}")
    print(f"Datasets compared: {len(dataset_avg_metrics)}")
    print("Strict statistics calculation to ensure consistency")
    print("Done!")
