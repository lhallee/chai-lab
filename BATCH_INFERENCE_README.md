# Batch Inference for Chai-1 Dimers

This script provides an efficient way to predict protein dimer structures and extract iPTM scores using the Chai-1 model. It loads the model weights once and reuses them for all predictions, making it much faster than running individual inference calls.

## Features

- **Efficient weight loading**: Model components are loaded once and cached for all predictions
- **iPTM score extraction**: Automatically extracts interface predicted TM (iPTM) scores for each dimer
- **CSV processing**: Processes CSV files with two columns representing protein sequences
- **No MSA requirement**: Runs without MSAs for faster inference as requested
- **Comprehensive scoring**: Outputs iPTM, pTM, and aggregate scores

## Usage

### Basic Usage

```bash
python batch_inference.py input.csv output.csv
```

### With Custom Device

```bash
python batch_inference.py input.csv output.csv --device cuda:0
```

## Input Format

The input CSV file should have exactly 2 columns with protein sequences. Column names can be anything (e.g., "protein_A", "protein_B").

Example `input.csv`:
```csv
protein_A,protein_B
MKLLTTTAAGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAAS,AIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKN
MKLLTTAAGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAAS,GAAL
```

## Output Format

The output CSV file will contain:
- `dimer_id`: Sequential ID for each dimer
- `sequence_1`: First protein sequence
- `sequence_2`: Second protein sequence  
- `iptm_score`: Interface predicted TM score (main metric of interest)
- `ptm_score`: Complex predicted TM score
- `aggregate_score`: Overall ranking score used by Chai-1

Example `output.csv`:
```csv
dimer_id,sequence_1,sequence_2,iptm_score,ptm_score,aggregate_score
0,MKLLTTTAAG...,AIQRTPKIQV...,0.7234,0.6891,0.7012
1,MKLLTTAAGS...,GAAL,0.4523,0.5678,0.4321
```

## Key Features for Efficiency

### 1. Component Caching
The script uses Chai-1's built-in component caching mechanism (`_component_cache`) to load model weights once:
- `feature_embedding.pt`
- `bond_loss_input_proj.pt`
- `token_embedder.pt`
- `trunk.pt`
- `diffusion_module.pt`
- `confidence_head.pt`

### 2. Optimized Settings
- Single diffusion sample (`num_diffn_samples=1`) for speed
- No MSA server usage (`use_msa_server=False`)
- No ESM embeddings (`use_esm_embeddings=False`)
- Low memory mode enabled (`low_memory=True`)

### 3. iPTM Score Extraction
The script directly extracts iPTM scores from the ranking results:
```python
iptm_score = best_ranking.ptm_scores.interface_ptm.item()
```

## Understanding iPTM Scores

The iPTM (interface predicted TM) score is a key metric for protein-protein interactions:
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: 
  - > 0.8: Very high confidence in the interface
  - 0.5-0.8: Moderate confidence
  - < 0.5: Low confidence
- **Definition**: Maximum TM score over chains, restricting to interactions between each chain and all other chains

## Performance Considerations

### Memory Usage
- Uses `low_memory=True` to minimize GPU memory usage
- Temporary files are cleaned up automatically
- Components are moved to CPU when not in use

### Speed Optimizations
- Model components are cached after first use
- Single sample prediction per dimer
- Minimal diffusion timesteps while maintaining quality

### Recommended Hardware
- GPU with at least 8GB VRAM (A100/H100 recommended)
- 16GB+ system RAM
- Fast storage for temporary files

## Example Workflow

1. Prepare your CSV file with two columns of protein sequences
2. Run the batch inference script
3. Analyze the iPTM scores in the output CSV
4. Use iPTM scores to filter high-confidence dimer predictions

```bash
# Example with the provided sample file
python batch_inference.py example_dimers.csv results.csv

# Check results
head -n 5 results.csv
```

## Error Handling

The script includes robust error handling:
- Failed predictions are marked with NaN values
- Detailed logging shows progress and any issues
- Continues processing even if individual dimers fail

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- pandas
- chai_lab package
- GPU with sufficient memory

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use smaller sequences
2. **Invalid sequences**: Ensure sequences contain only valid amino acid codes
3. **File permissions**: Ensure write permissions for output directory

### Performance Tips
1. Use sequences of similar length in batches for better GPU utilization
2. Monitor GPU memory usage during processing
3. Consider splitting very large CSV files into smaller chunks 