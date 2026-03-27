import os
import sys
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from InnoWhitespaceExtractor.models import Autoencoder
from zeroshot import init_zsinvert, zsinvert_decode

def main():
    print("=======================================================")
    print(" Phase 2: ZSInvert Storytelling (LLM)")
    print("=======================================================\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, 'outputs_neurips')
    
    vacancy_path = os.path.join(output_dir, '4_vacancy_results', 'vacancy_coordinates.csv')
    model_dir = os.path.join(output_dir, '2_autoencoder_model')
    
    if not os.path.exists(vacancy_path):
        print(f"Error: Vacancy file not found at {vacancy_path}. Run create_vacancy.py first.")
        return
        
    vacancies_df = pd.read_csv(vacancy_path)
    if vacancies_df.empty:
        print("No vacancies found to invert.")
        return
        
    # Filter only for Zone A
    target_zones = ['A', 'B', 'C']
    vacancies_df = vacancies_df[vacancies_df['zone'].str.upper().isin(target_zones)]
    
    if vacancies_df.empty:
        print("No vacancies found in Zone A to invert.")
        return
        
    print(f"Loaded {len(vacancies_df)} vacancies from Zone A for decoding.")
    
    models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not models:
        print("Error: Autoencoder model not found in {model_dir}")
        return
    model_path = os.path.join(model_dir, models[0])
    
    print(f"Loading Autoencoder: {models[0]}")
    ae = Autoencoder(embedding_dim=1024).double().to(device)
    ae.load_state_dict(torch.load(model_path, map_location=device))
    ae.eval()
    
    coords_tensor = torch.tensor(vacancies_df[['x', 'y']].values, dtype=torch.float64).to(device)
    with torch.no_grad():
        restored_embeddings = ae.decoder(coords_tensor)
        
    ae.to('cpu')
    
    restored_embeddings = restored_embeddings.float()
    
    print("\nLoading massive LLM and Encoder into VRAM... (This only happens once!)")
    attack_runner = init_zsinvert(device=device, llm_name='Qwen/Qwen2.5-7B-Instruct', encoder_name='thenlper/gte-large')

    results = []
    print("\nStarting ZSInvert generation loop...")
    for idx, emb in enumerate(restored_embeddings):
        print(f"\n--- Inverting Vacancy {idx+1}/{len(restored_embeddings)} ---")
        emb_target = emb.unsqueeze(0).to(device) # [1, 1024]
        
        # Inject structural Gaussian noise (jitter) to force varied generative paths for dense spatial clusters
        # Normalizes scale based on target standard deviation to be robust 
        noise = torch.randn_like(emb_target) * (0.05 * emb_target.std().item())
        emb_target_noisy = emb_target + noise
        
        try:
            gen_text, cos_sim = zsinvert_decode(
                attack=attack_runner,
                target_embedding=emb_target_noisy, 
                beam_width=5,
                max_steps=32
            )
            print(f"Result: {gen_text}")
            print(f"Sim: {cos_sim}")
        except Exception as e:
            print(f"Error during inversion: {e}")
            gen_text, cos_sim = "[Error]", 0.0
            
        results.append({
            'x': vacancies_df.iloc[idx]['x'],
            'y': vacancies_df.iloc[idx]['y'],
            'zone': vacancies_df.iloc[idx]['zone'] if 'zone' in vacancies_df.columns else 'N/A',
            'generated_text': gen_text,
            'cos_sim': float(cos_sim) if cos_sim else 0.0
        })
        
        # Save temp output in case of crash
        temp_df = pd.DataFrame(results)
        out_path = os.path.join(output_dir, '5_inversion_results', 'zsinvert_results.csv')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        temp_df.to_csv(out_path, index=False)
        
        # Free GPU memory between vacancies
        torch.cuda.empty_cache()
        
    print(f"\n[SUCCESS] Phase 2 finished. Results saved to {out_path}")

if __name__ == "__main__":
    main()
