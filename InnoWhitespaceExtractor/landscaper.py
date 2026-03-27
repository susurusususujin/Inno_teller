"""Landscaper module for InnoWhitespaceExtractor.

This module defines the InnoLandscaper class, which is responsible for:
1. Converting raw text data into embeddings.
3. Generating a 2D technological paper landscape map (Latent Space).
"""

import os
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .models import Autoencoder, EarlyStopping
from .utils import get_embeddings_openai, get_embeddings_gte


class InnoLandscaper:
    """Manages the creation of the technology landscape."""

    def __init__(self, output_dir: str, device: str = None) -> None:
        """Initializes the InnoLandscaper.

        Args:
            output_dir: Directory to save all outputs.
            device: 'cuda' or 'cpu'. If None, detects automatically.
        """
        self.output_dir = output_dir
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.embedding_dir = os.path.join(output_dir, '1_embedding')
        self.model_dir = os.path.join(output_dir, '2_autoencoder_model')
        self.map_dir = os.path.join(output_dir, '3_paper_map')
        
        os.makedirs(self.embedding_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.map_dir, exist_ok=True)

    def embed_data(self, data_path: str, text_column: str, api_key: str, 
                   model_prefix: str = 'lidar') -> str:
        """Generates embeddings for the given data file.
        
        Args:
            data_path: Path to the raw input file (.csv, .xlsx).
            text_column: Column name containing text to embed.
            api_key: OpenAI API key.
            model_prefix: Prefix for the output filename.

        Returns:
            Path to the saved embedding pickle file.
        """
        print(f"--- [InnoLandscaper] Step 1: Embedding Data ---")
        embedding_filename = f'{model_prefix}_embedding.pkl'
        output_path = os.path.join(self.embedding_dir, embedding_filename)

        if os.path.exists(output_path):
            print(f"  Skipping: Embeddings already exist at {output_path}")
            return output_path

        # Load Data
        ext = os.path.splitext(data_path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(data_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(data_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        print(f"  Loaded {len(df)} records from {data_path}")
        
        # Embed
        df[text_column] = df[text_column].fillna('')
        texts = df[text_column].tolist()
        
        print("  Generating embeddings (this may take a while)...")
        embeddings = get_embeddings_gte(texts, model_name='thenlper/gte-large', device=self.device.type)
        
        embedding_column = f'embedding_{text_column}'
        df[embedding_column] = embeddings.tolist()
        
        df.to_pickle(output_path)
        print(f"  Saved embeddings to {output_path}")
        return output_path

    def train_model(self, embedding_path: str, text_column: str, 
                    epochs: int = 300, 
                    lr: float = 0.001, batch_size: int = 512, 
                    model_prefix: str = 'lidar', force_retrain: bool = False) -> str:
        """Trains an Autoencoder model.
        
        Args:
            embedding_path: Path to the dataframe with embeddings.
            text_column: Text column name used to infer embedding column name.

            epochs: Max training epochs.
            lr: Learning rate.
            batch_size: Batch size.
            model_prefix: Prefix for model filename.

        Returns:
            Path to the saved model state dict.
        """
        print(f"--- [InnoLandscaper] Step 2: Training AE Model ---")
        embedding_column = f'embedding_{text_column}'
        
        raw_data_filename = os.path.splitext(os.path.basename(embedding_path))[0].replace('_embedding', '')
        training_date = datetime.now().strftime("%Y%m%d")
        
        model_filename = f"{model_prefix}_ae_{raw_data_filename}_{training_date}.pth"
            
        model_path = os.path.join(self.model_dir, model_filename)
        
        # Check if model exists (simple check, strictly following implementation plan request for new structure)
        # Note: Ideally we might check for *any* matching model, but for now we look for exact match or overwrite.
        # To be safe and efficient, we can check if it exists.
        if os.path.exists(model_path) and not force_retrain:
             print(f"  Skipping: Model already exists at {model_path}")
             return model_path
        elif force_retrain:
             print(f"  [Autoencoder Training] Force retrain enabled.")

        # Load and Prep Data
        df = pd.read_pickle(embedding_path)
        # Attempt to sort by date if it exists, for time-split validation
        # Sort by year or date if available
        if 'year' in df.columns:
            df = df.sort_values(by='year').reset_index(drop=True)
        elif 'publication_date' in df.columns:
            df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
            df = df.sort_values(by='publication_date').reset_index(drop=True)
        
        # User requested 9:1 split sorted by year
        train_size = int(0.9 * len(df))
        train_df = df[:train_size]
        val_df = df[train_size:]
        
        if len(train_df) == 0:
            train_df = df
            val_df = df
        
        # train_size = int(0.8 * len(df))
        # train_df = df[:train_size]
        # val_df = df[train_size:]
        
        # if len(train_df) == 0:
        #     train_df = df
        #     val_df = df

        train_tensor = torch.tensor(train_df[embedding_column].values.tolist(), dtype=torch.float64)
        val_tensor = torch.tensor(val_df[embedding_column].values.tolist(), dtype=torch.float64)
        
        print(f"  Training samples: {len(train_tensor)}, Validation samples: {len(val_tensor)}")

        # Initialize Model
        model = Autoencoder().double().to(self.device)
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)
        # User requested increasing patience to 100
        early_stopping = EarlyStopping(patience=100, verbose=True, path=model_path)
        
        # Training Loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for i in range(0, len(train_tensor), batch_size):
                batch = train_tensor[i:i+batch_size].to(self.device)
                optimizer.zero_grad()
                
                if True:
                    decoded, encoded = model(batch)
                    # Use Cosine Similarity loss: 1 - cos(x, y)
                    recon_loss = 1 - F.cosine_similarity(decoded, batch, dim=1).mean()
                    
                    # Topology Loss
                    topo_loss = self._topology_loss(batch, encoded)
                    
                    loss = recon_loss + 0.5 * topo_loss
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= (len(train_tensor) / batch_size) # approx mean batch loss

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i in range(0, len(val_tensor), batch_size):
                    batch = val_tensor[i:i+batch_size].to(self.device)
                    if True:
                        decoded, encoded = model(batch)
                        # Validation loss also using Cosine Similarity
                        recon_loss = 1 - F.cosine_similarity(decoded, batch, dim=1).mean()
                        topo_loss = self._topology_loss(batch, encoded)
                        loss = recon_loss + 0.5 * topo_loss
                    val_loss += loss.item()
            
            val_loss /= (len(val_tensor) / batch_size)

            print(f'  Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("  Early stopping triggered.")
                break
        
        print(f"  Model saved to {model_path}")
        return model_path



    def generate_map(self, embedding_path: str, model_path: str, 
                     text_column: str) -> str:
        """Generates 2D coordinates for the landscape.

        Args:
            embedding_path: Path to embedding pickle.
            model_path: Path to trained model.
            text_column: Text column name.


        Returns:
            Path to the saved coordinates CSV.
        """
        print(f"--- [InnoLandscaper] Step 3: Generating Map Coordinates ---")
        
        # Load Model
        model = Autoencoder().double().to(self.device)
             
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        # Load Data
        df = pd.read_pickle(embedding_path)
        embedding_column = f'embedding_{text_column}'
        embeddings = torch.tensor(df[embedding_column].values.tolist(), dtype=torch.float64)
        
        # Inference
        with torch.no_grad():
            _, coords_tensor = model(embeddings.to(self.device))
            coords = coords_tensor.cpu().numpy()
        
        # Save
        df_coords = pd.DataFrame(coords, columns=['X', 'Y'])
        coord_path = os.path.join(self.map_dir, 'paper_map_coordinates.csv')
        df_coords.to_csv(coord_path, index=False)
        print(f"  Saved 2D coordinates to {coord_path}")
        
        return coord_path

    def _topology_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Calculates topology preservation loss (distance correlation)."""
        # Pairwise euclidean distances
        # x: (B, D_in), z: (B, D_latent)
        
        dist_x = torch.cdist(x, x, p=2)
        dist_z = torch.cdist(z, z, p=2)
        
        # Normalize to be scale-invariant
        dist_x = dist_x / (dist_x.mean() + 1e-8)
        dist_z = dist_z / (dist_z.mean() + 1e-8)
        
        # MSE between normalized distance matrices
        loss = F.mse_loss(dist_x, dist_z)
        return loss