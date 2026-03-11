"""
Módulo de comparación de entrenamientos.
Genera gráficas profesionales para comparar N entrenamientos a partir de CSVs.

Uso:
    from utils.visualization.compare import comparar_entrenamientos

    # 2 o más archivos → gráficas comparativas
    comparar_entrenamientos(
        ('logs/training_history_v1.csv', 'Sin Transformer'),
        ('logs/training_history_v2.csv', 'Con Transformer'),
        ('logs/training_history_v3.csv', 'ConvNeXtV2 + T'),
    )

    # 1 solo archivo → gráficas individuales
    comparar_entrenamientos(
        ('logs/training_history_v1.csv', 'Mi modelo'),
    )
"""

import ast
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np


# Paleta de colores premium (hasta 8 modelos)
_COLORES = [
    '#6366f1',  # Índigo
    '#f43f5e',  # Rosa
    '#10b981',  # Esmeralda
    '#f59e0b',  # Ámbar
    '#06b6d4',  # Cyan
    '#8b5cf6',  # Violeta
    '#ec4899',  # Fucsia
    '#14b8a6',  # Teal
]

_MARCADORES = ['o', 's', 'D', '^', 'v', 'P', 'X', 'h']


def _parsear_iou_global(val):
    """Parsea 'tensor(0.78, device=...)' o un float directo."""
    if isinstance(val, (int, float)):
        return float(val)
    return float(str(val).split('(')[1].split(',')[0])


def _parsear_iou_clases(val):
    """Parsea un string de lista a lista de floats."""
    if isinstance(val, list):
        return val
    return ast.literal_eval(val)


def _cargar_csv(csv_path):
    """Carga un CSV de entrenamiento y parsea las columnas de IoU."""
    df = pd.read_csv(csv_path, comment='#')
    df['mIoU'] = df['val_iou Global'].apply(_parsear_iou_global)

    iou_clases = df['val_iou Clases'].apply(_parsear_iou_clases).apply(pd.Series)
    iou_clases.columns = ['Mascota', 'Fondo', 'Borde']

    return df, iou_clases


def _aplicar_estilo_eje(ax, titulo, xlabel='Epoch', ylabel=None):
    """Aplica estilos consistentes a un eje."""
    ax.set_title(titulo, fontsize=13, fontweight='bold', pad=12, color='#e2e8f0')
    ax.set_xlabel(xlabel, fontsize=10, color='#94a3b8')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color='#94a3b8')

    ax.grid(True, alpha=0.15, color='#475569', linestyle='--')
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8', labelsize=9)

    for spine in ax.spines.values():
        spine.set_color('#334155')
        spine.set_linewidth(0.8)

    ax.legend(
        fontsize=8,
        facecolor='#0f172a',
        edgecolor='#334155',
        labelcolor='#e2e8f0',
        loc='best',
        framealpha=0.9
    )


def _plot_metrica(ax, datos_lista, col_name, titulo, ylabel=None, formato_pct=False):
    """Plotea una métrica para N modelos."""
    for i, (df, _, nombre, epochs) in enumerate(datos_lista):
        color = _COLORES[i % len(_COLORES)]
        marker = _MARCADORES[i % len(_MARCADORES)]

        valores = df[col_name]
        ax.plot(
            epochs, valores,
            color=color, marker=marker, markersize=4, linewidth=2,
            label=nombre, alpha=0.9, markeredgecolor='white', markeredgewidth=0.5
        )

        # Marcar el mejor punto
        if 'loss' in col_name.lower():
            best_idx = valores.idxmin()
        else:
            best_idx = valores.idxmax()

        ax.annotate(
            f'{valores[best_idx]:.4f}' if not formato_pct else f'{valores[best_idx]:.2f}%',
            xy=(epochs[best_idx], valores[best_idx]),
            fontsize=7, color=color, fontweight='bold',
            textcoords="offset points", xytext=(0, 10),
            ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#0f172a', edgecolor=color, alpha=0.8)
        )

    _aplicar_estilo_eje(ax, titulo, ylabel=ylabel)


def _plot_iou_clase(ax, datos_lista, clase_nombre, titulo):
    """Plotea IoU de una clase específica para N modelos."""
    for i, (_, iou_df, nombre, epochs) in enumerate(datos_lista):
        color = _COLORES[i % len(_COLORES)]
        marker = _MARCADORES[i % len(_MARCADORES)]

        valores = iou_df[clase_nombre]
        ax.plot(
            epochs, valores,
            color=color, marker=marker, markersize=4, linewidth=2,
            label=nombre, alpha=0.9, markeredgecolor='white', markeredgewidth=0.5
        )

        best_idx = valores.idxmax()
        ax.annotate(
            f'{valores[best_idx]:.4f}',
            xy=(epochs[best_idx], valores[best_idx]),
            fontsize=7, color=color, fontweight='bold',
            textcoords="offset points", xytext=(0, 10),
            ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#0f172a', edgecolor=color, alpha=0.8)
        )

    _aplicar_estilo_eje(ax, titulo)


def _imprimir_tabla(datos_lista):
    """Imprime tabla resumen con los mejores resultados."""
    n = len(datos_lista)
    col_width = 18

    print("\n" + "=" * (28 + col_width * n))
    print("  📊 RESUMEN COMPARATIVO — Mejor epoch de cada modelo")
    print("=" * (28 + col_width * n))

    # Header
    header = f"{'Métrica':<26}"
    for _, _, nombre, _ in datos_lista:
        header += f"{nombre:>{col_width}}"
    print(header)
    print("-" * (26 + col_width * n))

    metricas = []
    for df, iou_df, nombre, _ in datos_lista:
        best = df['mIoU'].idxmax()
        metricas.append({
            'epoch': best + 1,
            'val_loss': df.loc[best, 'val_loss'],
            'val_acc': df.loc[best, 'val_acc'],
            'mIoU': df.loc[best, 'mIoU'],
            'mascota': iou_df.loc[best, 'Mascota'],
            'fondo': iou_df.loc[best, 'Fondo'],
            'borde': iou_df.loc[best, 'Borde'],
        })

    filas = [
        ('Mejor epoch',  'epoch',    '{:>'+str(col_width)+'d}'),
        ('Val Loss',     'val_loss', '{:>'+str(col_width)+'.4f}'),
        ('Val Accuracy', 'val_acc',  '{:>'+str(col_width - 1)+'.2f}%'),
        ('mIoU Global',  'mIoU',    '{:>'+str(col_width)+'.4f}'),
        ('IoU Mascota',  'mascota', '{:>'+str(col_width)+'.4f}'),
        ('IoU Fondo',    'fondo',   '{:>'+str(col_width)+'.4f}'),
        ('IoU Borde ⚠️', 'borde',   '{:>'+str(col_width)+'.4f}'),
    ]

    for label, key, fmt in filas:
        row = f"  {label:<24}"
        valores = [m[key] for m in metricas]
        best_val = max(valores) if key != 'val_loss' else min(valores)

        for val in valores:
            formatted = fmt.format(val)
            if val == best_val and n > 1:
                formatted = f"\033[92m{formatted}\033[0m"  # Verde para el mejor
            row += formatted
        print(row)

    print("=" * (28 + col_width * n))


def comparar_entrenamientos(*archivos):
    """
    Compara N entrenamientos con gráficas profesionales.

    Args:
        *archivos: Tuplas de (ruta_csv, nombre_modelo).
                   Ej: ('logs/train_v1.csv', 'Sin Transformer'),
                       ('logs/train_v2.csv', 'Con Transformer')

    Si se pasa un solo archivo, muestra gráficas individuales.
    Si se pasan 2+, muestra comparativas.
    """
    if not archivos:
        raise ValueError("Debes pasar al menos un archivo: comparar_entrenamientos(('ruta.csv', 'Nombre'), ...)")

    # Validar formato
    for arg in archivos:
        if not isinstance(arg, (tuple, list)) or len(arg) != 2:
            raise ValueError(
                f"Cada argumento debe ser (ruta_csv, nombre). Recibí: {arg}\n"
                f"Uso: comparar_entrenamientos(('archivo.csv', 'Nombre del modelo'), ...)"
            )

    # Cargar todos los CSVs
    datos_lista = []
    for csv_path, nombre in archivos:
        df, iou_df = _cargar_csv(csv_path)
        epochs = list(range(1, len(df) + 1))
        datos_lista.append((df, iou_df, nombre, epochs))

    # Configurar estilo global
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Segoe UI', 'Arial', 'DejaVu Sans'],
    })

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.set_facecolor('#0f172a')

    # Título principal
    n_modelos = len(datos_lista)
    titulo = 'Análisis de Entrenamiento' if n_modelos == 1 else f'Comparación de {n_modelos} Modelos'
    fig.suptitle(
        titulo,
        fontsize=18, fontweight='bold', color='#f1f5f9',
        y=0.98
    )

    # Subtítulo con nombres
    nombres_str = ' vs '.join([nombre for _, _, nombre, _ in datos_lista])
    fig.text(
        0.5, 0.94, nombres_str,
        ha='center', fontsize=11, color='#64748b', style='italic'
    )

    # Gráficas
    _plot_metrica(axes[0, 0], datos_lista, 'train_loss', '📉 Train Loss', ylabel='Loss')
    _plot_metrica(axes[0, 1], datos_lista, 'val_loss', '📉 Validation Loss', ylabel='Loss')
    _plot_metrica(axes[0, 2], datos_lista, 'val_acc', '🎯 Validation Accuracy', ylabel='%', formato_pct=True)
    _plot_metrica(axes[1, 0], datos_lista, 'mIoU', '📐 mIoU Global', ylabel='IoU')
    _plot_iou_clase(axes[1, 1], datos_lista, 'Mascota', '🐾 IoU Mascota')
    _plot_iou_clase(axes[1, 2], datos_lista, 'Borde', '⚠️ IoU Borde')

    plt.subplots_adjust(top=0.90, hspace=0.35, wspace=0.30, left=0.06, right=0.97, bottom=0.06)
    plt.show()

    # Tabla resumen en consola
    _imprimir_tabla(datos_lista)
