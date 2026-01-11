"""
Evolutionary Distance Calculation Pipeline

Calculates pairwise evolutionary distances between species using:
1. DNA sequences with Jukes-Cantor (JC69) correction
2. Protein sequences with Poisson correction
3. Codon-based alignment scores

Outputs distance matrices suitable for phylogenetic tree construction.

Authors:
  Wassan Haj Yahia
  Salah Mahmied
  Abed Jabaly
  Rawaa Aburaya
  Juman Abu Rmeleh
Course: Computational Biology
"""

import csv
import os
import sys
import glob
import math
from typing import Dict, List, Tuple
from Bio import SeqIO
import numpy as np

from sequence_alignment import (
    dna_alignment,
    protein_alignment,
    codon_alignment,
    add_stop_penalties,
    BLOSUM62_NO_STOP
)


# =============================================================================
# Substitution Rates
# =============================================================================

# These rates are based on published estimates for mammalian nuclear genes
DNA_SUBSTITUTION_RATE = 2.2e-9      # per site per year
PROTEIN_SUBSTITUTION_RATE = 1.0e-9  # proteins evolve slower due to selection


# =============================================================================
# File Discovery
# =============================================================================

def discover_data_files(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Automatically discover mRNA and protein FASTA files in a directory.
    Works with any number of species.
    """
    if not os.path.isdir(data_dir):
        print(f"[ERROR] Directory not found: {data_dir}")
        return [], []

    # Find all fasta files
    all_files = glob.glob(os.path.join(data_dir, "*.fasta"))

    mrna_files = []
    protein_files = []

    for f in all_files:
        basename = os.path.basename(f).lower()
        if basename.startswith("mrna_") or basename.startswith("mrna"):
            mrna_files.append(f)
        elif basename.startswith("protien_") or basename.startswith("protein_"):
            protein_files.append(f)

    # Sort for consistent ordering
    mrna_files.sort()
    protein_files.sort()

    return mrna_files, protein_files


# =============================================================================
# File Parsing
# =============================================================================

def parse_fasta_files(file_paths: List[str], sequence_type: str = "mRNA") -> Dict[str, str]:
    """Load sequences from FASTA files into a dictionary."""
    print(f"\n  Loading {sequence_type} sequences...")
    sequences = {}

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"    [!] File not found: {file_path}")
            continue

        species_name = os.path.basename(file_path).split('.')[0]
        for record in SeqIO.parse(file_path, "fasta"):
            sequence = str(record.seq).upper().replace('U', 'T')
            sequences[species_name] = sequence
            print(f"    {species_name}: {len(sequence)} {'bp' if sequence_type == 'mRNA' else 'aa'}")
            break

    return sequences


def find_species_key(sequences: Dict[str, str], search_term: str) -> str:
    """Find dictionary key containing search term."""
    search_lower = search_term.lower()
    for key in sequences.keys():
        if search_lower in key.lower():
            return key
    raise KeyError(f"Species '{search_term}' not found")


def match_mrna_protein_keys(mrna_keys: List[str], protein_sequences: Dict[str, str]) -> List[str]:
    """Match mRNA keys to corresponding protein keys."""
    protein_keys = []
    for mrna_key in mrna_keys:
        suffix = mrna_key.split('_', 1)[-1].lower()
        matched = None
        for prot_key in protein_sequences.keys():
            if prot_key.split('_', 1)[-1].lower() == suffix:
                matched = prot_key
                break
        if matched is None:
            raise KeyError(f"No protein match for {mrna_key}")
        protein_keys.append(matched)
    return protein_keys


# =============================================================================
# Distance Calculations
# =============================================================================

def calculate_p_distance(aligned_a: str, aligned_b: str) -> float:
    """
    Calculate p-distance from aligned sequences.
    p = mismatches / comparable_sites (excluding gaps)
    """
    if len(aligned_a) != len(aligned_b):
        raise ValueError("Aligned sequences must have equal length")

    mismatches = 0
    comparable = 0

    for a, b in zip(aligned_a, aligned_b):
        if a == '-' or b == '-':
            continue
        comparable += 1
        if a != b:
            mismatches += 1

    if comparable == 0:
        return float('nan')
    return mismatches / comparable


def jukes_cantor_correction(p: float) -> float:
    """
    Apply Jukes-Cantor correction for multiple substitutions.
    d = -3/4 * ln(1 - 4p/3)
    """
    if math.isnan(p) or p < 0:
        return float('nan')
    if p >= 0.75:
        return float('inf')
    try:
        return -0.75 * math.log(1 - (4.0 / 3.0) * p)
    except ValueError:
        return float('inf')


def poisson_correction(p: float) -> float:
    """
    Apply Poisson correction for protein sequences.
    d = -ln(1 - p)
    """
    if math.isnan(p) or p < 0:
        return float('nan')
    if p >= 1.0:
        return float('inf')
    try:
        return -math.log(1 - p)
    except ValueError:
        return float('inf')


def distance_to_mya(distance: float, rate: float) -> float:
    """Convert evolutionary distance to time in MYA using molecular clock."""
    if math.isnan(distance) or distance == float('inf'):
        return float('inf')
    return (distance / (2.0 * rate)) / 1e6


def convert_scores_to_distances(score_matrix: np.ndarray) -> np.ndarray:
    """Convert similarity scores to distances for clustering."""
    max_score = np.max(score_matrix)
    if max_score <= 0:
        return np.zeros_like(score_matrix)

    distance_matrix = max_score - score_matrix
    np.fill_diagonal(distance_matrix, 0)

    max_dist = np.max(distance_matrix)
    if max_dist > 0:
        distance_matrix = distance_matrix / max_dist * 100

    return distance_matrix


# =============================================================================
# Analysis Functions
# =============================================================================

def find_reference_species(mrna_seqs: Dict[str, str]) -> str:
    """Find human or first available species as reference."""
    # Try to find human first
    for key in mrna_seqs.keys():
        if "homo" in key.lower() or "sapiens" in key.lower():
            return key
    # Otherwise return first species
    return list(mrna_seqs.keys())[0]


def analyze_pairwise_comparisons(mrna_seqs: Dict[str, str], protein_seqs: Dict[str, str],
                                  blosum62: Dict, dna_rate: float, protein_rate: float):
    """Compare all species against reference species and display results."""

    print("\n" + "=" * 90)
    print("PAIRWISE COMPARISONS WITH REFERENCE")
    print("=" * 90)

    ref_mrna = find_reference_species(mrna_seqs)

    # Find matching protein key
    ref_suffix = ref_mrna.split('_', 1)[-1].lower()
    ref_prot = None
    for pk in protein_seqs.keys():
        if pk.split('_', 1)[-1].lower() == ref_suffix:
            ref_prot = pk
            break

    if ref_prot is None:
        print(f"[ERROR] No protein match for reference {ref_mrna}")
        return

    print(f"\nReference: {ref_mrna}")
    print(f"  mRNA: {len(mrna_seqs[ref_mrna])} bp | Protein: {len(protein_seqs[ref_prot])} aa")

    print("\n" + "-" * 90)
    print(f"{'Species':<28} {'DNA p-dist':>10} {'JC69 dist':>10} {'DNA (MYA)':>10} "
          f"{'Prot p-dist':>11} {'Poisson':>10} {'Prot (MYA)':>10}")
    print("-" * 90)

    for other_key in mrna_seqs.keys():
        if other_key == ref_mrna:
            continue

        # DNA analysis
        _, dna_aln_a, dna_aln_b = dna_alignment(
            mrna_seqs[ref_mrna], mrna_seqs[other_key], return_alignment=True)
        dna_p = calculate_p_distance(dna_aln_a, dna_aln_b)
        dna_d = jukes_cantor_correction(dna_p)
        dna_mya = distance_to_mya(dna_d, dna_rate)

        # Protein analysis
        other_suffix = other_key.split('_', 1)[-1].lower()
        other_prot = None
        for pk in protein_seqs.keys():
            if pk.split('_', 1)[-1].lower() == other_suffix:
                other_prot = pk
                break

        if other_prot is None:
            continue

        _, prot_aln_a, prot_aln_b = protein_alignment(
            protein_seqs[ref_prot], protein_seqs[other_prot], blosum62, return_alignment=True)
        prot_p = calculate_p_distance(prot_aln_a, prot_aln_b)
        prot_d = poisson_correction(prot_p)
        prot_mya = distance_to_mya(prot_d, protein_rate)

        # Format output
        species = other_key.replace('mRna_', '').replace('mRNA_', '')
        dna_p_s = f"{dna_p:.4f}" if not math.isnan(dna_p) else "N/A"
        dna_d_s = f"{dna_d:.4f}" if not math.isinf(dna_d) else "Sat."
        dna_mya_s = f"{dna_mya:.2f}" if not math.isinf(dna_mya) else "Sat."
        prot_p_s = f"{prot_p:.4f}" if not math.isnan(prot_p) else "N/A"
        prot_d_s = f"{prot_d:.4f}" if not math.isinf(prot_d) else "Sat."
        prot_mya_s = f"{prot_mya:.2f}" if not math.isinf(prot_mya) else "Sat."

        print(f"{species:<28} {dna_p_s:>10} {dna_d_s:>10} {dna_mya_s:>10} "
              f"{prot_p_s:>11} {prot_d_s:>10} {prot_mya_s:>10}")

    print("-" * 90)


def compute_distance_matrices(mrna_seqs: Dict[str, str], protein_seqs: Dict[str, str],
                               blosum62: Dict, dna_rate: float, protein_rate: float):
    """Compute all pairwise distance matrices."""

    print("\n" + "=" * 70)
    print("COMPUTING DISTANCE MATRICES")
    print("=" * 70)

    species_list = list(mrna_seqs.keys())
    n = len(species_list)
    protein_keys = match_mrna_protein_keys(species_list, protein_seqs)

    print(f"\n  Species count: {n}")
    print(f"  Pairwise comparisons: {n * (n - 1) // 2}")

    dna_matrix = np.zeros((n, n))
    protein_matrix = np.zeros((n, n))
    codon_scores = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # DNA
            _, dna_a, dna_b = dna_alignment(
                mrna_seqs[species_list[i]], mrna_seqs[species_list[j]], return_alignment=True)
            dna_p = calculate_p_distance(dna_a, dna_b)
            dna_d = jukes_cantor_correction(dna_p)
            dna_mya = distance_to_mya(dna_d, dna_rate)
            dna_matrix[i, j] = dna_matrix[j, i] = dna_mya

            # Protein
            _, prot_a, prot_b = protein_alignment(
                protein_seqs[protein_keys[i]], protein_seqs[protein_keys[j]],
                blosum62, return_alignment=True)
            prot_p = calculate_p_distance(prot_a, prot_b)
            prot_d = poisson_correction(prot_p)
            prot_mya = distance_to_mya(prot_d, protein_rate)
            protein_matrix[i, j] = protein_matrix[j, i] = prot_mya

            # Codon
            score = codon_alignment(mrna_seqs[species_list[i]], mrna_seqs[species_list[j]], blosum62)
            codon_scores[i, j] = codon_scores[j, i] = score

    codon_distances = convert_scores_to_distances(codon_scores)

    print("  Matrices computed successfully.")

    return species_list, dna_matrix, protein_matrix, codon_scores, codon_distances


def save_matrix(matrix: np.ndarray, species: List[str], filename: str):
    """Save matrix to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Species"] + species)
        for i, row in enumerate(matrix):
            writer.writerow([species[i]] + [f"{v:.6f}" for v in row])
    print(f"    Saved: {filename}")


def print_usage():
    """Print usage instructions."""
    print("""
Usage: python evolutionary_distance.py <data_directory>

Examples:
  python evolutionary_distance.py final_data/gapdh
  python evolutionary_distance.py final_data/PRDM9
  python evolutionary_distance.py final_data/ssh3

The directory should contain:
  - mRna_*.fasta files (mRNA sequences)
  - protien_*.fasta files (protein sequences)

Files will be automatically discovered.
""")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("EVOLUTIONARY DISTANCE ANALYSIS")
    print("Assessing Relationships Using DNA, Protein, and Codon Alignments")
    print("=" * 70)

    # Check command line arguments
    if len(sys.argv) < 2:
        print("\n[ERROR] No data directory specified.")
        print_usage()

        # Default to gapdh if no argument provided
        print("Using default: final_data/gapdh")
        data_dir = "final_data/gapdh"
    else:
        data_dir = sys.argv[1]

    # Get gene name from directory
    gene_name = os.path.basename(os.path.normpath(data_dir)).upper()
    print(f"\nAnalyzing gene: {gene_name}")
    print(f"Data directory: {data_dir}")

    # Discover files automatically
    print("\n[1] FILE DISCOVERY")
    mrna_files, protein_files = discover_data_files(data_dir)

    if len(mrna_files) == 0:
        print(f"\n[ERROR] No mRNA files found in {data_dir}")
        print("Expected files like: mRna_Homo_sapiens.fasta")
        exit(1)

    if len(protein_files) == 0:
        print(f"\n[ERROR] No protein files found in {data_dir}")
        print("Expected files like: protien_Homo_sapiens.fasta")
        exit(1)

    print(f"  Found {len(mrna_files)} mRNA files")
    print(f"  Found {len(protein_files)} protein files")

    # Setup
    print("\n[2] INITIALIZATION")
    BLOSUM62 = add_stop_penalties(dict(BLOSUM62_NO_STOP), stop_penalty=-5)
    print(f"  BLOSUM62 matrix loaded ({len(BLOSUM62)} entries)")

    # Load data
    print("\n[3] DATA LOADING")
    mrna_seqs = parse_fasta_files(mrna_files, "mRNA")
    protein_seqs = parse_fasta_files(protein_files, "Protein")

    if len(mrna_seqs) == 0:
        print("\n[ERROR] No sequences loaded. Check file paths.")
        exit(1)

    # Analysis
    print("\n[4] PAIRWISE ANALYSIS")
    analyze_pairwise_comparisons(mrna_seqs, protein_seqs, BLOSUM62,
                                  DNA_SUBSTITUTION_RATE, PROTEIN_SUBSTITUTION_RATE)

    print("\n[5] MATRIX COMPUTATION")
    species, dna_mat, prot_mat, codon_scores, codon_dist = compute_distance_matrices(
        mrna_seqs, protein_seqs, BLOSUM62, DNA_SUBSTITUTION_RATE, PROTEIN_SUBSTITUTION_RATE)

    # Save results
    print("\n[6] SAVING RESULTS")
    save_matrix(dna_mat, species, "dna_distance_matrix.csv")
    save_matrix(prot_mat, species, "protein_distance_matrix.csv")
    save_matrix(codon_scores, species, "codon_score_matrix.csv")
    save_matrix(codon_dist, species, "codon_distance_matrix.csv")

    # Summary
    print("\n" + "=" * 70)
    print(f"ANALYSIS COMPLETE - {gene_name}")
    print("=" * 70)
    print(f"\nGene: {gene_name}")
    print(f"Species analyzed: {len(species)}")
    print("\nOutput Files:")
    print("  - dna_distance_matrix.csv      (Jukes-Cantor corrected)")
    print("  - protein_distance_matrix.csv  (Poisson corrected)")
    print("  - codon_score_matrix.csv       (Raw alignment scores)")
    print("  - codon_distance_matrix.csv    (Converted for clustering)")
    print("\nMethodology:")
    print("  1. Needleman-Wunsch global alignment")
    print("  2. p-distance from aligned sequences")
    print("  3. Evolutionary corrections (JC69 / Poisson)")
    print("  4. Molecular clock: T = d / (2r)")
    print("=" * 70 + "\n")