#!/usr/bin/env python3
"""
Batch inference script for Chai-1 model to predict dimer structures and extract iPTM scores.

This script loads the model weights once and processes a CSV file with two columns
representing protein sequences for dimers. It extracts the iPTM score for each dimer
and saves results to an output CSV. Optionally saves all output files when save_cifs is enabled.

Usage:
    python batch_inference.py input.csv output.csv
    python batch_inference.py input.csv output.csv --save_cifs --output_dir results/
"""

import argparse
import pandas as pd
import torch
import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, Optional
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


def run_single_inference(seq1: str, seq2: str, idx: int, device: torch.device, 
                        save_cifs: bool = False, output_base_dir: Optional[Path] = None) -> Dict[str, float]:
    """Run inference on a single dimer and extract iPTM score."""
    
    if save_cifs and output_base_dir:
        # Use persistent directory for saving outputs
        output_dir = output_base_dir / f"dimer_{idx}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create FASTA file in the output directory
        fasta_content = create_fasta_string(seq1, seq2, idx)
        fasta_path = output_dir / f"dimer_{idx}.fasta"
        fasta_path.write_text(fasta_content)
        
        use_temp_dir = False
        temp_dir_context = None
    else:
        # Use temporary directory (original behavior)
        temp_dir_context = tempfile.TemporaryDirectory()
        temp_dir = Path(temp_dir_context.name)
        
        # Create FASTA file
        fasta_content = create_fasta_string(seq1, seq2, idx)
        fasta_path = temp_dir / f"dimer_{idx}.fasta"
        fasta_path.write_text(fasta_content)
        
        # Create output directory
        output_dir = temp_dir / f"output_{idx}"
        output_dir.mkdir()
        use_temp_dir = True
    
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
        
        result = {
            'iptm': iptm_score,
            'ptm': ptm_score,
            'aggregate_score': aggregate_score
        }
        
        # Log output directory if saving files
        if save_cifs:
            logging.info(f"Dimer {idx} outputs saved to: {output_dir}")
        
        return result
        
    except Exception as e:
        logging.error(f"Inference failed for dimer {idx}: {e}")
        return {
            'iptm': float('nan'),
            'ptm': float('nan'),
            'aggregate_score': float('nan')
        }
    finally:
        # Clean up temporary directory if used
        if use_temp_dir and temp_dir_context:
            temp_dir_context.cleanup()


def process_csv(input_csv: str, output_csv: str, device: torch.device, 
               save_cifs: bool = False, output_dir: Optional[str] = None) -> None:
    """Process CSV file with dimers and extract iPTM scores."""
    logging.info(f"Reading input CSV: {input_csv}")
    
    # Read input CSV
    df = pd.read_csv(input_csv)
    
    if df.shape[1] != 2:
        raise ValueError(f"Input CSV must have exactly 2 columns, got {df.shape[1]}")
    
    # Get column names
    col1, col2 = df.columns[0], df.columns[1]
    logging.info(f"Processing {len(df)} dimers from columns: {col1}, {col2}")
    
    # Setup output directory if saving CIFs
    output_base_dir = None
    if save_cifs:
        if output_dir is None:
            output_dir = "batch_inference_outputs"
        output_base_dir = Path(output_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving outputs to: {output_base_dir.absolute()}")
    
    # Pre-load model components
    preload_model_components(device)
    
    # Process each dimer
    results = []
    for idx, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df), desc="Processing dimers"):
        seq1 = str(row[col1])
        seq2 = str(row[col2])
        
        logging.info(f"Processing dimer {idx + 1}/{len(df)}")
        
        # Run inference
        scores = run_single_inference(seq1, seq2, idx, device, save_cifs, output_base_dir)
        
        # Store results
        result = {
            'dimer_id': idx,
            'sequence_1': seq1,
            'sequence_2': seq2,
            'iptm_score': scores['iptm'],
            'ptm_score': scores['ptm'],
            'aggregate_score': scores['aggregate_score']
        }
        
        # Add output path if saving files
        if save_cifs and output_base_dir:
            result['output_path'] = str(output_base_dir / f"dimer_{idx}")
        
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
    parser.add_argument('--save_cifs', action='store_true', help='Save all output files (CIFs, etc.) instead of using temporary files')
    parser.add_argument('--output_dir', type=str, default='batch_inference_outputs', help='Directory to save output files when --save_cifs is enabled')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_logging()
    logging.info(f"Using device: {device}")
    
    if args.save_cifs:
        logging.info(f"Will save all output files to: {args.output_dir}")
    else:
        logging.info("Using temporary files (outputs will be deleted after processing)")
    
    # Process CSV
    process_csv(args.input_csv, args.output_csv, device, args.save_cifs, args.output_dir)


if __name__ == "__main__":
    main() 