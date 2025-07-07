#!/usr/bin/env python3
"""
Batch inference script for Chai-1 model to predict dimer structures and extract iPTM scores.

This script loads the model weights once and processes a CSV file with two columns
representing protein sequences for dimers. It extracts the iPTM score for each dimer
and saves results to an output CSV.

Usage:
    python batch_inference.py input.csv output.csv
"""

import argparse
import pandas as pd
import torch
import logging
import tempfile
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from chai_lab.chai1 import make_all_atom_feature_context, run_folding_on_context


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_fasta_string(seq1: str, seq2: str, idx: int) -> str:
    """Create a FASTA string for a dimer with two protein sequences."""
    return f""">protein|chain_A_{idx}
{seq1}
>protein|chain_B_{idx}
{seq2}"""


def preload_model_components(device: torch.device) -> None:
    """Pre-load all model components to cache them for efficient inference."""
    logging.info("Pre-loading model components...")
    
    # Create a minimal dummy feature context to trigger model loading
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_fasta = Path(tmp_dir) / "dummy.fasta"
        dummy_fasta.write_text(">protein|dummy\nMKLLTT")
        
        dummy_output_dir = Path(tmp_dir) / "dummy_output"
        dummy_output_dir.mkdir()
        
        # This will load and cache all model components
        try:
            feature_context = make_all_atom_feature_context(
                fasta_file=dummy_fasta,
                output_dir=dummy_output_dir,
                use_esm_embeddings=True,
                use_msa_server=False,
                msa_directory=None,
                constraint_path=None,
                use_templates_server=False,
                templates_path=None,
                esm_device=device,
            )
            
            # Run a minimal inference to load all components
            _ = run_folding_on_context(
                feature_context,
                output_dir=dummy_output_dir,
                num_trunk_recycles=1,  # Minimal recycles
                num_diffn_timesteps=5,  # Minimal timesteps
                num_diffn_samples=1,   # Single sample
                seed=42,
                device=device,
                low_memory=False,
            )
            
        except Exception as e:
            logging.warning(f"Model pre-loading failed: {e}")
            logging.info("Continuing without pre-loading - components will load on first use")
    
    logging.info("Model components pre-loaded successfully")


def run_single_inference(seq1: str, seq2: str, idx: int, device: torch.device) -> Dict[str, float]:
    """Run inference on a single dimer and extract iPTM score."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create FASTA file
        fasta_content = create_fasta_string(seq1, seq2, idx)
        fasta_path = Path(tmp_dir) / f"dimer_{idx}.fasta"
        fasta_path.write_text(fasta_content)
        
        # Create output directory
        output_dir = Path(tmp_dir) / f"output_{idx}"
        output_dir.mkdir()
        
        try:
            # Create feature context
            feature_context = make_all_atom_feature_context(
                fasta_file=fasta_path,
                output_dir=output_dir,
                use_esm_embeddings=True,
                use_msa_server=False,
                msa_directory=None,
                constraint_path=None,
                use_templates_server=False,
                templates_path=None,
                esm_device=device,
            )
            
            # Run inference with minimal samples for efficiency
            candidates = run_folding_on_context(
                feature_context,
                output_dir=output_dir,
                num_trunk_recycles=3,
                num_diffn_timesteps=200,
                num_diffn_samples=1,  # Single sample for efficiency
                seed=42,
                device=device,
                low_memory=False,
            )
            
            # Extract iPTM score from the best candidate
            best_ranking = candidates.ranking_data[0]
            iptm_score = best_ranking.ptm_scores.interface_ptm.item()
            
            # Also extract other useful scores
            ptm_score = best_ranking.ptm_scores.complex_ptm.item()
            aggregate_score = best_ranking.aggregate_score.item()
            
            return {
                'iptm': iptm_score,
                'ptm': ptm_score,
                'aggregate_score': aggregate_score
            }
            
        except Exception as e:
            logging.error(f"Inference failed for dimer {idx}: {e}")
            return {
                'iptm': float('nan'),
                'ptm': float('nan'),
                'aggregate_score': float('nan')
            }


def process_csv(input_csv: str, output_csv: str, device: torch.device) -> None:
    """Process CSV file with dimers and extract iPTM scores."""
    logging.info(f"Reading input CSV: {input_csv}")
    
    # Read input CSV
    df = pd.read_csv(input_csv)
    
    if df.shape[1] != 2:
        raise ValueError(f"Input CSV must have exactly 2 columns, got {df.shape[1]}")
    
    # Get column names
    col1, col2 = df.columns[0], df.columns[1]
    logging.info(f"Processing {len(df)} dimers from columns: {col1}, {col2}")
    
    # Pre-load model components
    preload_model_components(device)
    
    # Process each dimer
    results = []
    for idx, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df), desc="Processing dimers"):
        seq1 = str(row[col1])
        seq2 = str(row[col2])
        
        logging.info(f"Processing dimer {idx + 1}/{len(df)}")
        
        # Run inference
        scores = run_single_inference(seq1, seq2, idx, device)
        
        # Store results
        result = {
            'dimer_id': idx,
            'sequence_1': seq1,
            'sequence_2': seq2,
            'iptm_score': scores['iptm'],
            'ptm_score': scores['ptm'],
            'aggregate_score': scores['aggregate_score']
        }
        results.append(result)
        
        logging.info(f"Dimer {idx + 1}: iPTM={scores['iptm']:.4f}, pTM={scores['ptm']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    logging.info(f"Results saved to: {output_csv}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Batch inference for Chai-1 dimers')
    parser.add_argument('--input_csv', type=str, default='example_dimers.csv', help='Input CSV file with two columns for protein sequences')
    parser.add_argument('--output_csv', type=str, default='results.csv', help='Output CSV file for results')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_logging()
    logging.info(f"Using device: {device}")
    
    # Process CSV
    process_csv(args.input_csv, args.output_csv, device)


if __name__ == "__main__":
    main() 