import os
import sys
import torch

# Add adversarial_decoding repo root to path so we can import 'adversarial_decoding'
base_dir = os.path.abspath(os.path.dirname(__file__))
adv_dir = os.path.join(base_dir, 'adversarial_decoding')
if adv_dir not in sys.path:
    sys.path.insert(0, adv_dir)

from sentence_transformers import SentenceTransformer
from adversarial_decoding.strategies.emb_inv_decoding import EmbInvDecoding

def init_zsinvert(device='cuda',
                  llm_name='Qwen/Qwen2.5-7B-Instruct',
                  encoder_name='thenlper/gte-large'):
    """Initializes and returns the EmbInvDecoding attack instance and loads models to VRAM."""
    print(f"[zeroshot] Loading encoder {encoder_name} on {device}...")
    encoder = SentenceTransformer(encoder_name).to(device)
    
    print(f"[zeroshot] Initializing ZSInvert attack strategy with LLM {llm_name}...")
    attack = EmbInvDecoding(
        encoder=encoder,
        device=device,
        should_natural=False,
        repetition_penalty=1.5,
        model_name=llm_name
    )
    return attack

def zsinvert_decode(attack: EmbInvDecoding,
                    target_embedding: torch.Tensor,
                    beam_width=10, 
                    max_steps=32, 
                    top_k=20, 
                    top_p=1.0):
    """
    Decodes a target embedding back to text using the pre-initialized ZSInvert attack object.
    target_embedding: [1, D] shaped embedding tensor
    """
    # print(f"[zeroshot] Starting beam search decoding...") # suppressed to reduce log noise
    best_cand = attack.run_decoding(
        prompt='English research paper abstract:',
        target=target_embedding,
        beam_width=beam_width,
        max_steps=max_steps,
        top_k=top_k,
        top_p=top_p,
        should_full_sent=False,
        verbose=False,
        randomness=True
    )
    
    # Return best text and final cos_sim
    return best_cand.seq_str, best_cand.cos_sim
