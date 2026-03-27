import os
import sys

# Add inner module paths if necessary, assuming script runs from business_engine
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from InnoWhitespaceExtractor.landscaper import InnoLandscaper
from InnoWhitespaceExtractor.teller import InnoTeller

def main():
    print("=======================================================")
    print(" Phase 1: Vacancy Extraction (gte-large)")
    print("=======================================================\n")
    
    # Configuration
    base_dir = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, 'outputs_neurips')
    data_path = os.path.join(base_dir, 'neurips_data_clean.csv')
    text_column = 'abstract'
    model_prefix = 'neurips_gte'
    
    # Initialize
    landscaper = InnoLandscaper(output_dir=output_dir)
    # teller only used for density math and coordinate finding, no openai key needed now
    teller = InnoTeller(output_dir=output_dir, api_key=None)

    print("\n>>> Step 1-3: Landscaping (Embed -> Train -> Map)")
    # We pass a dummy api_key because utils.py's gte-large won't even use it
    embedding_path = landscaper.embed_data(
        data_path=data_path, 
        text_column=text_column, 
        api_key="sk-dummy",
        model_prefix=model_prefix
    )
    
    model_path = landscaper.train_model(
        embedding_path=embedding_path,
        text_column=text_column,
        epochs=300,
        lr=0.001,
        batch_size=512,
        model_prefix=model_prefix,
        force_retrain=True
    )
    
    coords_path = landscaper.generate_map(
        embedding_path=embedding_path,
        model_path=model_path,
        text_column=text_column
    )

    print("\n>>> Step 4: Vacancy Identification")
    vacancy_path = teller.identify_vacancies(
        coords_path=coords_path,
        embedding_path=embedding_path,
        grid_size='auto',
        density_batch_size=1024
    )
    
    print("\n[SUCCESS] Phase 1 finished.")
    print(f"Vacancies saved at: {vacancy_path}")
    print(f"AE Model saved at : {model_path}")
    print("\nNext, run Phase 2 (ZSinvert) to translate these vacancies to text.")

if __name__ == "__main__":
    main()
