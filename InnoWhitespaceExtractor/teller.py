"""Teller module for InnoWhitespaceExtractor.

This module defines the InnoTeller class, which is responsible for:
1. Identifying vacancies (white spaces) in the technology map.
2. Inverting these vacancies back to the embedding space.
3. Converting the inverted embeddings into human-readable text descriptions.
"""

import os
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from .models import Autoencoder
from .utils import (
    get_freedman_diaconis_bins,
    calculate_kde_bandwidth,
    create_grid,
    calculate_density_optimized,
    plot_density_contours
)


class InnoTeller:
    """Manages the identification and textual description of technology vacancies."""

    def __init__(self, output_dir: str, device: str = None, api_key: str = None) -> None:
        """Initializes the InnoTeller.

        Args:
            output_dir: Directory to save all outputs.
            device: 'cuda' or 'cpu'.
            api_key: OpenAI API key (needed for vec2text).
        """
        self.output_dir = output_dir
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.api_key = api_key
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        self.map_dir = os.path.join(output_dir, '3_paper_map')
        self.vacancy_dir = os.path.join(output_dir, '4_vacancy_results')
        self.inversion_dir = os.path.join(output_dir, '5_inversion_results')
        
        os.makedirs(self.vacancy_dir, exist_ok=True)
        os.makedirs(self.inversion_dir, exist_ok=True)

    def identify_vacancies(self, coords_path: str, embedding_path: str, 
                           grid_size: str = 'auto', density_batch_size: int = 1024) -> str:
        """Identifies vacancies in the paper map.

        Args:
            coords_path: Path to the 2D coordinates CSV.
            embedding_path: Path to the original embeddings (used for plotting/reference).
            grid_size: 'auto' or float.
            density_batch_size: Batch size for KDE.

        Returns:
            Path to the vacancy coordinates CSV.
        """
        print(f"--- [InnoTeller] Step 4: Identifying Vacancies ---")
        
        # Load Coordinates
        df_coords = pd.read_csv(coords_path)
        two_dim_coords = df_coords[['X', 'Y']].values
        
        # Determine Grid Size
        if grid_size == 'auto':
            num_bins = get_freedman_diaconis_bins(df_coords['X'])
            calculated_grid_size = (df_coords['X'].max() - df_coords['X'].min()) / num_bins
            
            min_sensible = (df_coords['X'].max() - df_coords['X'].min()) / 200
            final_grid_size = max(calculated_grid_size, min_sensible)
            
            print(f"  Auto-calculated grid size: {final_grid_size:.4f}")
            final_grid_size /= 3.0
            print(f"  Adjusted grid size: {final_grid_size:.4f}")
        else:
            final_grid_size = float(grid_size)

        # Calculate Density
        bandwidth = calculate_kde_bandwidth(two_dim_coords)
        x_grid, y_grid = create_grid(two_dim_coords, final_grid_size)
        
        print(f"  Calculating density on grid shape {x_grid.shape} (Adaptive KDE)...")
        grid_density = calculate_density_optimized(
            two_dim_coords, x_grid, y_grid, bandwidth, self.device, density_batch_size, adaptive=True
        )
        
        # Find Vacancies by Zone
        all_vacancies = []
        zone_summary = []
        all_vacancies_by_zone = {}
        zone_names = ['A', 'B', 'C', 'D', 'E', 'F']
        
        for zone_name in zone_names:
            vacancies = self._find_vacancy_points(
                grid_density, x_grid, y_grid, two_dim_coords, final_grid_size, zone_name
            )
            all_vacancies_by_zone[zone_name] = vacancies
            zone_summary.append({'zone': zone_name, 'vacancy_count': len(vacancies)})
            for vac in vacancies:
                all_vacancies.append({'x': vac[0], 'y': vac[1], 'zone': zone_name})
        
        # Decoupled Plotting: Save data to npy
        np.save(os.path.join(self.map_dir, 'x_grid.npy'), x_grid)
        np.save(os.path.join(self.map_dir, 'y_grid.npy'), y_grid)
        np.save(os.path.join(self.map_dir, 'grid_density.npy'), grid_density)
        np.save(os.path.join(self.map_dir, 'two_dim_coords.npy'), two_dim_coords)
        
        # Save Results
        vacancy_df = pd.DataFrame(all_vacancies)
        vacancy_path = os.path.join(self.vacancy_dir, 'vacancy_coordinates.csv')
        vacancy_df.to_csv(vacancy_path, index=False)
        
        summary_df = pd.DataFrame(zone_summary)
        summary_path = os.path.join(self.vacancy_dir, 'vacancy_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        print(f"  Identified {len(vacancy_df)} total vacancies. Saved to {vacancy_path}")
        return vacancy_path

    def _find_vacancy_points(self, grid_density, x_grid, y_grid, data, grid_size, selected_zone):
        density_flat = grid_density.ravel()
        percentiles_indices = [0, 50, 75, 90, 95, 99, 100]
        levels = np.unique(np.percentile(density_flat, percentiles_indices))
        
        zone_map = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1}
        
        # Safety check if fewer levels than zones (e.g. uniform distribution)
        # Original code logic: zone_index maps to levels index
        idx_map = {v: k for k, v in zone_map.items()} # 6->A, 5->B...
        
        # Map selected zone to level index.
        # Levels has size up to 7. indices 0..6.
        # A (Top 1%) -> 99-100 percentile -> Last interval -> index -2 to -1?
        # Original code: zone_index = zone_map(A) - 1 = 5. levels[5] to levels[6].
        
        desired_level_idx = zone_map.get(selected_zone.upper()) - 1
        
        if desired_level_idx < 0 or desired_level_idx + 1 >= len(levels):
            return []
            
        min_density = levels[desired_level_idx]
        max_density = levels[desired_level_idx + 1]
        
        if min_density == max_density:
            max_density += 1e-9

        # Mask grid points in this density range
        mask = (grid_density >= min_density) & (grid_density < max_density)
        potential_points = np.column_stack((x_grid[mask], y_grid[mask]))
        
        if not len(potential_points):
            return []
            
        # Filter points that are NOT close to real data
        tree = cKDTree(data)
        indices = tree.query_ball_point(potential_points, r=grid_size / 2)
        
        # Ind is list of neighbors. data point is a vacancy if NO neighbors within r.
        vacancy_points = [potential_points[i] for i, ind in enumerate(indices) if not ind]
        
        return vacancy_points

    def tell_story(self, vacancy_path: str, model_path: str, zone: str, 
                   num_steps: int = 50, 
                   sequence_beam_width: int = 3) -> str:
        """Inverts vacancies to text (tells the story of the whitespace).
        
        Args:
            vacancy_path: Path to vacancy coordinates CSV.
            model_path: Path to trained model.
            zone: Target zone ('A' or 'ALL').
            num_steps: Steps for vec2text.
            sequence_beam_width: Beam width for vec2text.
            
        Returns:
            Path to the inversion results CSV.
        """
        print(f"--- [InnoTeller] Step 5: Inverting Vacancies (Storytelling) ---")
        
        try:
            all_vacancies_df = pd.read_csv(vacancy_path)
            if all_vacancies_df.empty:
                print("  Vacancy file is empty.")
                return None
        except Exception:
            print("  Vacancy file not found or empty.")
            return None

        # Filter Zone
        if zone.upper() == 'ALL':
            vacancies_df = all_vacancies_df
        else:
            vacancies_df = all_vacancies_df[all_vacancies_df['zone'].str.upper() == zone.upper()].copy()

        if vacancies_df.empty:
            print(f"  No vacancies found for zone {zone}.")
            return None
        
        print(f"  Processing {len(vacancies_df)} vacancies for Zone {zone}")

        # Load Model
        model = Autoencoder().double().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        # Decode Coordinates -> Embeddings
        coords_tensor = torch.tensor(vacancies_df[['x', 'y']].values, dtype=torch.float64).to(self.device)
        
        with torch.no_grad():
            restored_embeddings = model.decoder(coords_tensor)
        
        # Free GPU for vec2text if needed
        model.to('cpu')
        torch.cuda.empty_cache()

        # Invert Embeddings -> Text
        print("  Loading vec2text corrector (text-embedding-ada-002)...")
        import vec2text
        corrector = vec2text.load_pretrained_corrector("text-embedding-ada-002")
        
        generated_texts = []
        batch_size = 2 # Small batch for vec2text to avoid OOM or API rate limits?
        
        # Convert to float32 for vec2text compatibility
        embedding_tensor_float = restored_embeddings.float().cpu()
        
        total = len(embedding_tensor_float)
        print(f"  Inverting {total} embeddings...")
        
        for i in range(0, total, batch_size):
            batch = embedding_tensor_float[i:i+batch_size].to(self.device) # Move back to GPU for vec2text?
            # vec2text expects tensor on the same device as corrector? usually yes.
            # Assuming corrector is on GPU if available.
            
            try:
                batch_texts = vec2text.invert_embeddings(
                    embeddings=batch,
                    corrector=corrector,
                    num_steps=num_steps,
                    sequence_beam_width=sequence_beam_width
                )
                generated_texts.extend(batch_texts)
            except Exception as e:
                print(f"  Error in batch {i}: {e}")
                generated_texts.extend(["[Error]"] * len(batch))
            
            if (i+1) % 10 == 0:
                print(f"    Processed {i+1}/{total}")

        # Save Results
        results_df = pd.DataFrame({
            'x': vacancies_df['x'],
            'y': vacancies_df['y'],
            'zone': vacancies_df['zone'],
            'generated_abstract': generated_texts
        })
        
        output_filename = f"inversion_results_zone_{zone.lower()}.csv"
        output_path = os.path.join(self.inversion_dir, output_filename)
        results_df.to_csv(output_path, index=False)
        
        print(f"  Saved generated stories to {output_path}")
        return output_path