import os
import numpy as np
import pandas as pd
from InnoWhitespaceExtractor.utils import plot_density_contours

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    map_dir = os.path.join(base_dir, 'outputs_neurips', '3_paper_map')
    vacancy_path = os.path.join(base_dir, 'outputs_neurips', '4_vacancy_results', 'vacancy_coordinates.csv')
    
    # Check if necessary files exist
    if not os.path.exists(os.path.join(map_dir, 'grid_density.npy')):
        print("에러: 지형도 데이터를 찾을 수 없습니다. 먼저 create_vacancy.py를 실행하세요.")
        return

    print("--- [Visualization] Loading Map Data ---")
    x_grid = np.load(os.path.join(map_dir, 'x_grid.npy'))
    y_grid = np.load(os.path.join(map_dir, 'y_grid.npy'))
    grid_density = np.load(os.path.join(map_dir, 'grid_density.npy'))
    two_dim_coords = np.load(os.path.join(map_dir, 'two_dim_coords.npy'))
    
    # Recover grid_size from x_grid
    if x_grid.shape[1] > 1:
        grid_size = x_grid[0, 1] - x_grid[0, 0]
    else:
        grid_size = 0.015
    
    all_vacancies_by_zone = {z: [] for z in ['A', 'B', 'C', 'D', 'E', 'F']}
    
    if os.path.exists(vacancy_path):
        print(f"  Loading vacancies from {vacancy_path}")
        vac_df = pd.read_csv(vacancy_path)
        for _, row in vac_df.iterrows():
            zone = row['zone'].upper()
            if zone in all_vacancies_by_zone:
                all_vacancies_by_zone[zone].append([row['x'], row['y']])
    else:
        print("  No vacancy coordinates found. Drawing clean map only.")
        
    print("--- [Visualization] Plotting Paper Map ---")
    map_output_prefix = os.path.join(map_dir, 'paper_map')
    plot_density_contours(
        x_grid, y_grid, grid_density, grid_size, two_dim_coords, 
        all_vacancies_by_zone, map_output_prefix
    )
    print("--- [Visualization] Done! Map images are updated! ---")

if __name__ == "__main__":
    main()
