import argparse
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure the parent directory's src folder is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from inno_whitespace import InnoLandscaper, InnoTeller

def main():
    parser = argparse.ArgumentParser(description="InnoWhitespaceExtractor: Automated Vacancy Identification System")

    # Core Arguments
    parser.add_argument('--claim', action='store_true', help='Use claim data instead of abstract data.')
    parser.add_argument('--api_key', type=str, default=os.environ.get("OPENAI_API_KEY"), help='OpenAI API Key.')

    # Training/Model Params
    parser.add_argument('--epochs', type=int, default=300, help='Max epochs for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=512, help='Training batch size.')
    
    # Identification/Inversion Params
    parser.add_argument('--grid_size', type=str, default='auto', help='Grid size (auto or float).')
    parser.add_argument('--density_batch_size', type=int, default=1024, help='KDE batch size.')
    parser.add_argument('--zone', type=str, default='A', help='Target zone for inversion.')
    parser.add_argument('--num_steps', type=int, default=50, help='vec2text inversion steps.')
    parser.add_argument('--sequence_beam_width', type=int, default=3, help='vec2text beam width.')

    # Paths and Override Arguments
    parser.add_argument('--data_path', type=str, default=None, help='Raw data path (override default).')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (override default).')
    parser.add_argument('--model_prefix', type=str, default=None, help='Model file prefix (override default).')
    parser.add_argument('--text_column', type=str, default=None, help='Text column name (override default).')

    args = parser.parse_args()

    # --- Configuration ---
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Defaults
    if args.claim:
        default_output_dir = os.path.join(base_dir, 'outputs_claim_refactored')
        default_model_prefix = 'lidar_claim'
        default_text_column = 'claim'
        default_data_path = os.path.join(base_dir, 'data', 'all_lidar_claim.xlsx')
    else:
        default_output_dir = os.path.join(base_dir, 'outputs_refactored') # Changed to avoid collision with old runs
        default_model_prefix = 'lidar'
        default_text_column = 'abstract'
        default_data_path = os.path.join(base_dir, 'data', 'all_lidar.xlsx')

    # Apply Overrides
    output_dir = args.output_dir if args.output_dir else default_output_dir
    model_prefix = args.model_prefix if args.model_prefix else default_model_prefix
    text_column = args.text_column if args.text_column else default_text_column
    data_path = args.data_path if args.data_path else default_data_path

    print("=======================================================")
    print(f" InnoWhitespaceExtractor Pipeline")
    print(f" Data: {data_path}")
    print(f" Output: {output_dir}")
    print(f" Model Prefix: {model_prefix}")
    print("=======================================================\n")

    # --- Execution ---
    
    # 1. Initialize Modules
    landscaper = InnoLandscaper(output_dir=output_dir)
    teller = InnoTeller(output_dir=output_dir, api_key=args.api_key)

    # 2. Landscaper: Embedding -> Training -> Map
    print("\n>>> Phase 1: Landscaping")
    
    try:
        embedding_path = landscaper.embed_data(
            data_path=data_path, 
            text_column=text_column, 
            api_key=args.api_key,
            model_prefix=model_prefix
        )
        
        model_path = landscaper.train_model(
            embedding_path=embedding_path,
            text_column=text_column,
            epochs=args.epochs,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            model_prefix=model_prefix
        )
        
        coords_path = landscaper.generate_map(
            embedding_path=embedding_path,
            model_path=model_path,
            text_column=text_column
        )
    except Exception as e:
        print(f"\n[ERROR] Landscaping failed: {e}")
        return

    # 3. Teller: Vacancy Identification -> Storytelling (Inversion)
    print("\n>>> Phase 2: Storytelling")
    
    try:
        vacancy_path = teller.identify_vacancies(
            coords_path=coords_path,
            embedding_path=embedding_path,
            grid_size=args.grid_size,
            density_batch_size=args.density_batch_size
        )
        
        # Only proceed if vacancies were found
        teller.tell_story(
            vacancy_path=vacancy_path,
            model_path=model_path,
            zone=args.zone,
            num_steps=args.num_steps,
            sequence_beam_width=args.sequence_beam_width
        )
    except Exception as e:
        print(f"\n[ERROR] Storytelling failed: {e}")
        return

    print("\n[SUCCESS] InnoWhitespaceExtractor pipeline finished.")

if __name__ == "__main__":
    main()