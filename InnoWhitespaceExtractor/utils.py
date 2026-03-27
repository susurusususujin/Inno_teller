"""Utility module for InnoWhitespaceExtractor.

This module provides shared helper functions for:
- Text embedding with OpenAI (and tiktoken truncation).
- Kernel Density Estimation (KDE) and Grid generation.
- Visualization of density maps.
"""

import math
import os
from typing import List, Tuple, Dict, Optional, Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import openai
import tiktoken
from scipy.stats import iqr


# --- Embedding Utilities ---

def truncate_text_to_tokens(text: str, model: str, max_tokens: int) -> str:
    """Truncates a string to a maximum number of tokens.

    Args:
        text: The input string.
        model: The OpenAI model name (e.g., 'text-embedding-ada-002').
        max_tokens: Maximum allowed tokens.

    Returns:
        The truncated string.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    return text


def get_embeddings_openai(
    text_list: List[str],
    model: str = "text-embedding-ada-002",
    api_key: Optional[str] = None
) -> torch.Tensor:
    """Generates embeddings for a list of texts using OpenAI API.

    Args:
        text_list: List of strings to embed.
        model: Model identifier.
        api_key: OpenAI API key.

    Returns:
        A torch Tensor of embeddings (shape: [N, D]).
    """
    if api_key:
        openai.api_key = api_key

    batch_size = 64
    batches = math.ceil(len(text_list) / batch_size)
    outputs = []
    print(f"Total batches for embedding: {batches}")

    # ada-002 limit is 8192, use 8191 for safety
    MAX_TOKENS = 8191

    for batch in range(batches):
        if (batch + 1) % 10 == 0:
             print(f"  Processing batch {batch + 1}/{batches}")
        
        text_list_batch = text_list[batch * batch_size : (batch + 1) * batch_size]
        processed_batch = []
        for text in text_list_batch:
            # Handle None or non-string
            clean_text = str(text) if text is not None and not pd.isna(text) else ""
            if not clean_text.strip():
                clean_text = " "
            truncated_text = truncate_text_to_tokens(clean_text, model, MAX_TOKENS)
            processed_batch.append(truncated_text)

        try:
            response = openai.embeddings.create(
                input=processed_batch,
                model=model,
                encoding_format="float",
            )
            batch_embeddings = [e.embedding for e in response.data]
            outputs.extend(batch_embeddings)
        except Exception as e:
            print(f"Error in batch {batch}: {e}")
            # Identify which text caused error or retry logic could go here
            # For now, append zeros or raise
            raise e

    return torch.tensor(outputs)


def get_embeddings_gte(
    text_list: List[str],
    model_name: str = "thenlper/gte-large",
    device: str = "cuda"
) -> torch.Tensor:
    """Generates embeddings for a list of texts using local Hugging Face model."""
    from sentence_transformers import SentenceTransformer
    if not hasattr(get_embeddings_gte, "model"):
        print(f"Loading {model_name} on {device}...")
        get_embeddings_gte.model = SentenceTransformer(model_name).to(device)
    
    print(f"Generating embeddings for {len(text_list)} records...")
    
    processed_batch = []
    for text in text_list:
        clean_text = str(text) if text is not None and not pd.isna(text) else ""
        if not clean_text.strip():
            clean_text = " "
        processed_batch.append(clean_text)

    embeddings = get_embeddings_gte.model.encode(processed_batch, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    return embeddings.cpu()


# --- Density & Grid Utilities ---

def get_freedman_diaconis_bins(data: np.ndarray) -> int:
    """Calculates bin count using Freedman-Diaconis rule."""
    q75, q25 = np.percentile(data, [75 ,25])
    iqr_val = q75 - q25
    if iqr_val == 0:
        return 50 # Default if IQR is 0
    
    h = 2 * iqr_val / (len(data) ** (1/3))
    if h == 0:
        return 50
    return int(np.ceil((data.max() - data.min()) / h))


def calculate_kde_bandwidth(data: np.ndarray) -> np.ndarray:
    """Calculates KDE bandwidth using Scott's Rule."""
    n = len(data)
    sigma = np.std(data, axis=0)
    bandwidth = (n ** (-1 / 6.0)) * sigma
    return bandwidth


def create_grid(
    data: np.ndarray, grid_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a meshgrid covering the data range."""
    x_min, x_max = data[:, 0].min() - grid_size, data[:, 0].max() + grid_size
    y_min, y_max = data[:, 1].min() - grid_size, data[:, 1].max() + grid_size
    
    x_centers = np.arange(x_min, x_max, grid_size) + grid_size / 2
    y_centers = np.arange(y_min, y_max, grid_size) + grid_size / 2
    
    if len(x_centers) == 0 or len(y_centers) == 0:
         raise ValueError("Grid size is too large for the data range.")
         
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    return x_grid, y_grid


def calculate_density_optimized(
    data: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    bandwidth: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
    adaptive: bool = True
) -> np.ndarray:
    """Calculates Gaussian KDE density on a grid using PyTorch (Adaptive or Fixed)."""
    data_t = torch.tensor(data, device=device, dtype=torch.float64)
    x_grid_t = torch.tensor(x_grid, device=device, dtype=torch.float64)
    y_grid_t = torch.tensor(y_grid, device=device, dtype=torch.float64)
    
    n = data_t.shape[0]
    grid_points = torch.stack([x_grid_t.ravel(), y_grid_t.ravel()], dim=1)
    
    # --- Adaptive Bandwidth Logic ---
    if adaptive:
        # 1. Pilot Estimate (using fixed bandwidth)
        print("    [Adaptive KDE] Calculating pilot density...")
        pilot_bandwidth_t = torch.tensor(bandwidth, device=device, dtype=torch.float64)
        bw0, bw1 = pilot_bandwidth_t[0], pilot_bandwidth_t[1]
        norm_const_pilot = (n * (2 * np.pi * bw0 * bw1) ** 0.5)
        
        # We need pilot density AT THE DATA POINTS
        pilot_density_at_data = torch.zeros(n, device=device, dtype=torch.float64)
        
        # Calculate data-to-data density (pilot)
        # Using smaller batches to safely handle N*N matrix
        pilot_batch_size = 512
        for i in range(0, n, pilot_batch_size):
            batch_data = data_t[i:i+pilot_batch_size]
            # diff shape: (N_all, N_batch, 2)
            diff = (data_t.unsqueeze(1) - batch_data.unsqueeze(0)) / pilot_bandwidth_t.reshape(1, 1, -1)
            kernel_values = torch.exp(-0.5 * torch.sum(diff**2, dim=2))
            pilot_density_at_data[i:i+pilot_batch_size] = kernel_values.sum(dim=0) / norm_const_pilot

        # 2. Calculate Local Bandwidth Factors (lambda)
        # Geometric mean of pilot density
        g = torch.exp(torch.mean(torch.log(pilot_density_at_data + 1e-10)))
        alpha = 0.5 # Sensitivity parameter (0.5 is standard)
        local_lambdas = (pilot_density_at_data / g) ** -alpha
        
        # Clip lambdas to prevent extreme values (e.g., 0.1 to 5.0)
        local_lambdas = torch.clamp(local_lambdas, 0.1, 5.0)
        
        # 3. Apply Factors to Base Bandwidth
        # shape: (N, 2)
        adaptive_bandwidths = pilot_bandwidth_t.unsqueeze(0) * local_lambdas.unsqueeze(1)
        print(f"    [Adaptive KDE] Bandwidth range factors: {local_lambdas.min():.2f}x ~ {local_lambdas.max():.2f}x")
        
    else:
        adaptive_bandwidths = torch.tensor(bandwidth, device=device, dtype=torch.float64).unsqueeze(0).repeat(n, 1)

    # --- Final Density Calculation ---
    density = torch.zeros(grid_points.shape[0], device=device, dtype=torch.float64)
    
    # Pre-calculate normalization constant for each data point (since bandwidth varies per point)
    # bw shape: (N, 2)
    # norm_const shape: (N,)
    norm_consts = n * 2 * np.pi * adaptive_bandwidths[:, 0] * adaptive_bandwidths[:, 1]
    norm_consts = torch.sqrt(norm_consts) # sqrt of the product term? No, formula is (2pi * hx * hy)^(1/2) * something?
    # Correct Normalization for 2D Gaussian: 1 / (2 * pi * hx * hy * n) if independent?
    # Standard Multivariate Gaussian: 1 / ( sqrt((2pi)^k * |Sigma|) )
    # Here Sigma = diag(hx^2, hy^2). |Sigma| = hx^2 * hy^2. 
    # So denom = sqrt( (2pi)^2 * hx^2 * hy^2 ) = 2pi * hx * hy. 
    # Multiply by n for KDE average.
    norm_consts = n * 2 * np.pi * adaptive_bandwidths[:, 0] * adaptive_bandwidths[:, 1]
    
    # To optimize shape for broadcasting: (N, 1, 1)
    adaptive_bandwidths = adaptive_bandwidths.unsqueeze(1) # (N, 1, 2)
    norm_consts = norm_consts.unsqueeze(1) # (N, 1)

    for i in range(0, grid_points.shape[0], batch_size):
        batch_grid_points = grid_points[i : i + batch_size]
        
        # diff: (N_data, 1, 2) - (1, Batch, 2) -> (N_data, Batch, 2)
        diff = (data_t.unsqueeze(1) - batch_grid_points.unsqueeze(0)) / adaptive_bandwidths
        
        # kernel values: (N_data, Batch)
        sq_diff = torch.sum(diff**2, dim=2)
        kernel_values = torch.exp(-0.5 * sq_diff)
        
        # Divide by per-point normalization constant
        weighted_kernels = kernel_values / norm_consts
        
        # Sum over all data points to get density at each grid point
        batch_density = weighted_kernels.sum(dim=0)
        density[i : i + batch_size] = batch_density
        
    return density.cpu().numpy().reshape(x_grid.shape)


# --- Visualization Utilities ---

def plot_density_contours(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    grid_density: np.ndarray,
    grid_size: float,
    data: np.ndarray,
    all_vacancies_by_zone: Dict[str, List[Tuple[float, float]]],
    output_path_prefix: str
) -> None:
    """Plots and saves density contours and vacancies."""
    
    def create_base_plot():
        fig, ax = plt.subplots(figsize=(18, 14))
        density_flat = grid_density.ravel()
        percentiles_indices = [0, 50, 75, 90, 95, 99, 100]
        levels = np.unique(np.percentile(density_flat, percentiles_indices))
        
        colors = ['#2a56c6', '#7f99d0', '#b8c8e6', '#f5c1aa', '#e7865f', '#d73014']
        
        # Handle edge case where flat density is constant
        if len(levels) <= 1:
             levels = np.linspace(density_flat.min(), density_flat.max() + 1e-9, 2)

        contour_set = ax.contourf(x_grid, y_grid, grid_density, levels=levels, colors=colors)
        
        cbar = plt.colorbar(contour_set)
        cbar.set_ticks(levels)
        cbar.set_ticklabels([f'{p}%' for p in percentiles_indices[:len(levels)]], fontsize=15)
        
        custom_labels = ['F', 'E', 'D', 'C', 'B', 'A']
        for i, label in enumerate(custom_labels):
            if i + 1 < len(levels):
                position = levels[i] + (levels[i+1] - levels[i]) / 2
                cbar.ax.text(1.75, position, label, ha='center', va='center', fontsize=30, weight='bold')
                
        ax.plot(data[:, 0], data[:, 1], 'w.', markersize=2, alpha=0.7)
        
        # Draw the actual analysis grid
        # x_grid and y_grid contain centers. We want to draw lines at the edges.
        x_edges = x_grid[0, :] - grid_size / 2
        y_edges = y_grid[:, 0] - grid_size / 2
        # Add the last edge
        x_edges = np.append(x_edges, x_edges[-1] + grid_size)
        y_edges = np.append(y_edges, y_edges[-1] + grid_size)
        
        for x in x_edges:
            ax.axvline(x, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
        for y in y_edges:
            ax.axhline(y, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
            
        # Remove axis labels and ticks as requested
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Determine plot limits
        x_plot_min, x_plot_max = x_grid.min() - grid_size, x_grid.max() + grid_size
        y_plot_min, y_plot_max = y_grid.min() - grid_size, y_grid.max() + grid_size
        
        ax.set_xlim(x_plot_min, x_plot_max)
        ax.set_ylim(y_plot_min, y_plot_max)

        return fig, ax

    # 1. Plot individual zones
    for zone_name, vacancies in all_vacancies_by_zone.items():
        if not vacancies:
            continue
        fig, ax = create_base_plot()
        vac_x, vac_y = zip(*vacancies)
        ax.plot(vac_x, vac_y, 'kx', markersize=8, markeredgewidth=1.5, label=f'Zone {zone_name} Vacancies')
        ax.set_title(f'Paper Map - Vacancies in Zone {zone_name}', fontsize=20)
        ax.legend()
        
        path = f"{output_path_prefix}_zone_{zone_name}.png"
        plt.savefig(path)
        plt.close(fig)
        print(f"  Saved map for Zone {zone_name} to {path}")

    # 2. Plot clean map (no vacancies shown)
    fig, ax = create_base_plot()
    ax.set_title(f'Paper Map', fontsize=20)
    
    # User requested NO vacancies on the main map
    # for zone_name, vacancies in all_vacancies_by_zone.items():
    #     if vacancies:
    #         vac_x, vac_y = zip(*vacancies)
    #         ax.plot(vac_x, vac_y, 'x', markersize=6, label=f'Zone {zone_name}')
            
    # if all_vacancies_by_zone:
    #     ax.legend()
        
    # Save as paper_map.png
    clean_path = f"{output_path_prefix}.png"
    plt.savefig(clean_path)
    plt.close(fig)
    print(f"  Saved clean map to {clean_path}")