"""
Sequence Alignment Algorithms for Evolutionary Analysis

Implements three alignment methods for phylogenetic analysis:
1. DNA alignment - Needleman-Wunsch on nucleotide sequences
2. Protein alignment - Needleman-Wunsch using BLOSUM62 scoring matrix
3. Codon alignment - Frameshift-aware alignment at the codon level

All alignment functions support traceback to return aligned sequences,
which is required for calculating p-distances.

Authors:
  Wassan Haj Yahia
  Salah Mahmied
  Abed Jabaly
  Rawaa Aburaya
  Juman Abu Rmeleh
Course: Computational Biology
"""

import numpy as np
from typing import Dict, Tuple, Union

# =============================================================================
# Scoring Parameters
# =============================================================================

# DNA alignment scores
GAP_PENALTY_DNA = -2
MATCH_SCORE_DNA = 1
MISMATCH_SCORE_DNA = -1

# Codon alignment penalties
FRAME_SHIFT_PENALTY = -10
CODON_GAP_PENALTY = -6

# Standard genetic code
CODON_TABLE: Dict[str, str] = {
    "TAA": "*", "TAG": "*", "TGA": "*",
    "TTT": "F", "TTC": "F",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I",
    "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y",
    "CAT": "H", "CAC": "H",
    "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C",
    "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# BLOSUM62 substitution matrix
BLOSUM62_NO_STOP: Dict[Tuple[str, str], int] = {
    ('W', 'F'): 1, ('L', 'R'): -2, ('S', 'P'): -1, ('V', 'T'): 0,
    ('Q', 'Q'): 5, ('N', 'A'): -2, ('Z', 'Y'): -2, ('W', 'R'): -3,
    ('Q', 'A'): -1, ('S', 'D'): 0, ('H', 'H'): 8, ('S', 'H'): -1,
    ('H', 'D'): -1, ('L', 'N'): -3, ('W', 'A'): -3, ('Y', 'M'): -1,
    ('G', 'R'): -2, ('Y', 'I'): -1, ('Y', 'E'): -2, ('B', 'Y'): -3,
    ('Y', 'A'): -2, ('V', 'D'): -3, ('B', 'S'): 0, ('Y', 'Y'): 7,
    ('G', 'N'): 0, ('E', 'C'): -4, ('Y', 'Q'): -1, ('Z', 'Z'): 4,
    ('V', 'A'): 0, ('C', 'C'): 9, ('M', 'R'): -1, ('V', 'E'): -2,
    ('T', 'N'): 0, ('P', 'P'): 7, ('V', 'I'): 3, ('V', 'S'): -2,
    ('Z', 'P'): -1, ('V', 'M'): 1, ('T', 'F'): -2, ('V', 'Q'): -2,
    ('K', 'K'): 5, ('P', 'D'): -1, ('I', 'H'): -3, ('I', 'D'): -3,
    ('T', 'R'): -1, ('P', 'L'): -3, ('K', 'G'): -2, ('M', 'N'): -2,
    ('P', 'H'): -2, ('F', 'Q'): -3, ('Z', 'G'): -2, ('X', 'L'): -1,
    ('T', 'M'): -1, ('Z', 'C'): -3, ('X', 'H'): -1, ('D', 'R'): -2,
    ('B', 'W'): -4, ('X', 'D'): -1, ('Z', 'K'): 1, ('F', 'A'): -2,
    ('Z', 'W'): -3, ('F', 'E'): -3, ('D', 'N'): 1, ('B', 'K'): 0,
    ('X', 'X'): -1, ('F', 'I'): 0, ('B', 'G'): -1, ('X', 'T'): 0,
    ('F', 'M'): 0, ('B', 'C'): -3, ('Z', 'I'): -3, ('Z', 'V'): -2,
    ('S', 'S'): 4, ('L', 'Q'): -2, ('W', 'E'): -3, ('Q', 'R'): 1,
    ('N', 'N'): 6, ('W', 'M'): -1, ('Q', 'C'): -3, ('W', 'I'): -3,
    ('S', 'C'): -1, ('L', 'A'): -1, ('S', 'G'): 0, ('L', 'E'): -3,
    ('W', 'Q'): -2, ('H', 'G'): -2, ('S', 'K'): 0, ('Q', 'N'): 0,
    ('N', 'R'): 0, ('H', 'C'): -3, ('Y', 'N'): -2, ('G', 'Q'): -2,
    ('Y', 'F'): 3, ('C', 'A'): 0, ('V', 'L'): 1, ('G', 'E'): -2,
    ('G', 'A'): 0, ('K', 'R'): 2, ('E', 'D'): 2, ('Y', 'R'): -2,
    ('M', 'Q'): 0, ('T', 'I'): -1, ('C', 'D'): -3, ('V', 'F'): -1,
    ('T', 'A'): 0, ('T', 'P'): -1, ('B', 'P'): -2, ('T', 'E'): -1,
    ('V', 'N'): -3, ('P', 'G'): -2, ('M', 'A'): -1, ('K', 'H'): -1,
    ('V', 'R'): -3, ('P', 'C'): -3, ('M', 'E'): -2, ('K', 'L'): -2,
    ('V', 'V'): 4, ('M', 'I'): 1, ('T', 'Q'): -1, ('I', 'G'): -4,
    ('P', 'K'): -1, ('M', 'M'): 5, ('K', 'D'): -1, ('I', 'C'): -1,
    ('Z', 'D'): 1, ('F', 'R'): -3, ('X', 'K'): -1, ('Q', 'D'): 0,
    ('X', 'G'): -1, ('Z', 'L'): -3, ('X', 'C'): -2, ('Z', 'H'): 0,
    ('B', 'L'): -4, ('B', 'H'): 0, ('F', 'F'): 6, ('X', 'W'): -2,
    ('B', 'D'): 4, ('D', 'A'): -2, ('S', 'L'): -2, ('X', 'S'): 0,
    ('F', 'N'): -3, ('S', 'R'): -1, ('W', 'D'): -4, ('V', 'Y'): -1,
    ('W', 'L'): -2, ('H', 'R'): 0, ('W', 'H'): -2, ('H', 'N'): 1,
    ('W', 'T'): -2, ('T', 'T'): 5, ('S', 'F'): -2, ('W', 'P'): -4,
    ('L', 'D'): -4, ('B', 'I'): -3, ('L', 'H'): -3, ('S', 'N'): 1,
    ('B', 'T'): -1, ('L', 'L'): 4, ('Y', 'K'): -2, ('E', 'Q'): 2,
    ('Y', 'G'): -3, ('Z', 'S'): 0, ('Y', 'C'): -2, ('G', 'D'): -1,
    ('B', 'V'): -3, ('E', 'A'): -1, ('Y', 'W'): 2, ('E', 'E'): 5,
    ('Y', 'S'): -2, ('C', 'N'): -3, ('V', 'C'): -1, ('T', 'H'): -2,
    ('P', 'R'): -2, ('V', 'G'): -3, ('T', 'L'): -1, ('V', 'K'): -2,
    ('K', 'Q'): 1, ('R', 'A'): -1, ('I', 'R'): -3, ('T', 'D'): -1,
    ('P', 'F'): -4, ('I', 'N'): -3, ('K', 'I'): -3, ('M', 'D'): -3,
    ('V', 'W'): -3, ('W', 'W'): 11, ('M', 'H'): -2, ('P', 'N'): -2,
    ('K', 'A'): -1, ('M', 'L'): 2, ('K', 'E'): 1, ('Z', 'E'): 4,
    ('X', 'N'): -1, ('Z', 'A'): -1, ('Z', 'M'): -1, ('X', 'F'): -1,
    ('K', 'C'): -3, ('B', 'Q'): 0, ('X', 'B'): -1, ('B', 'M'): -3,
    ('F', 'C'): -2, ('Z', 'Q'): 3, ('X', 'Z'): -1, ('F', 'G'): -3,
    ('B', 'E'): 1, ('X', 'V'): -1, ('F', 'K'): -3, ('B', 'A'): -2,
    ('X', 'R'): -1, ('D', 'D'): 6, ('W', 'G'): -2, ('Z', 'F'): -3,
    ('S', 'Q'): 0, ('W', 'C'): -2, ('W', 'K'): -3, ('H', 'Q'): 0,
    ('L', 'C'): -1, ('W', 'N'): -4, ('S', 'A'): 1, ('L', 'G'): -4,
    ('W', 'S'): -3, ('S', 'E'): 0, ('H', 'E'): 0, ('S', 'I'): -2,
    ('H', 'A'): -2, ('S', 'M'): -1, ('Y', 'L'): -1, ('Y', 'H'): 2,
    ('Y', 'D'): -3, ('E', 'R'): 0, ('X', 'P'): -2, ('G', 'G'): 6,
    ('G', 'C'): -3, ('E', 'N'): 0, ('Y', 'T'): -2, ('Y', 'P'): -3,
    ('T', 'K'): -1, ('A', 'A'): 4, ('P', 'Q'): -1, ('T', 'C'): -1,
    ('V', 'H'): -3, ('T', 'G'): -2, ('I', 'Q'): -3, ('Z', 'T'): -1,
    ('C', 'R'): -3, ('V', 'P'): -2, ('P', 'E'): -1, ('M', 'C'): -1,
    ('K', 'N'): 0, ('I', 'I'): 4, ('P', 'A'): -1, ('M', 'G'): -3,
    ('T', 'S'): 1, ('I', 'E'): -3, ('P', 'M'): -2, ('M', 'K'): -1,
    ('I', 'A'): -1, ('P', 'I'): -3, ('R', 'R'): 5, ('X', 'M'): -1,
    ('L', 'I'): 2, ('X', 'I'): -1, ('Z', 'B'): 1, ('X', 'E'): -1,
    ('Z', 'N'): 0, ('X', 'A'): 0, ('B', 'R'): -1, ('B', 'N'): 3,
    ('F', 'D'): -3, ('X', 'Y'): -1, ('Z', 'R'): 0, ('F', 'H'): -1,
    ('B', 'F'): -3, ('F', 'L'): 0, ('X', 'Q'): -1, ('B', 'B'): 4
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_substitution_score(substitution_matrix: Dict[Tuple[str, str], int],
                           aa_a: str, aa_b: str, default: int = -1) -> int:
    """Look up substitution score, checking both orderings."""
    if (aa_a, aa_b) in substitution_matrix:
        return substitution_matrix[(aa_a, aa_b)]
    if (aa_b, aa_a) in substitution_matrix:
        return substitution_matrix[(aa_b, aa_a)]
    return default


def add_stop_penalties(substitution_matrix: Dict[Tuple[str, str], int],
                       stop_penalty: int = -5) -> Dict[Tuple[str, str], int]:
    """Add stop codon entries to the substitution matrix."""
    new_matrix = dict(substitution_matrix)
    all_symbols = set()
    for (a, b) in substitution_matrix.keys():
        all_symbols.add(a)
        all_symbols.add(b)
    for symbol in all_symbols:
        new_matrix[('*', symbol)] = stop_penalty
        new_matrix[(symbol, '*')] = stop_penalty
    new_matrix[('*', '*')] = stop_penalty
    return new_matrix


def translate_dna_to_protein(dna_seq: str) -> str:
    """Translate DNA sequence to protein."""
    dna_seq = dna_seq.upper().replace('U', 'T')
    num_codons = len(dna_seq) // 3
    protein = []
    for i in range(num_codons):
        codon = dna_seq[i*3:(i+1)*3]
        if '-' in codon:
            protein.append('-')
        else:
            protein.append(CODON_TABLE.get(codon, 'X'))
    return "".join(protein)


# =============================================================================
# DNA Alignment (Needleman-Wunsch)
# =============================================================================

def dna_alignment(seq_a: str, seq_b: str,
                  return_alignment: bool = False) -> Union[float, Tuple[float, str, str]]:
    """
    Global DNA alignment using Needleman-Wunsch.

    Scoring: Match +1, Mismatch -1, Gap -2
    """
    seq_a = seq_a.upper().replace('U', 'T')
    seq_b = seq_b.upper().replace('U', 'T')
    n, m = len(seq_a), len(seq_b)

    # DP and traceback matrices
    dp = np.zeros((n + 1, m + 1), dtype=float)
    traceback = np.zeros((n + 1, m + 1), dtype=np.int8)

    # Initialize borders
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + GAP_PENALTY_DNA
        traceback[i, 0] = 1
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + GAP_PENALTY_DNA
        traceback[0, j] = 2

    # Fill matrix
    for i in range(1, n + 1):
        char_a = seq_a[i - 1]
        for j in range(1, m + 1):
            char_b = seq_b[j - 1]

            if char_a == char_b:
                match_score = dp[i - 1, j - 1] + MATCH_SCORE_DNA
            else:
                match_score = dp[i - 1, j - 1] + MISMATCH_SCORE_DNA

            up_score = dp[i - 1, j] + GAP_PENALTY_DNA
            left_score = dp[i, j - 1] + GAP_PENALTY_DNA

            best_score = match_score
            best_move = 0
            if up_score > best_score:
                best_score = up_score
                best_move = 1
            if left_score > best_score:
                best_score = left_score
                best_move = 2

            dp[i, j] = best_score
            traceback[i, j] = best_move

    final_score = float(dp[n, m])

    if not return_alignment:
        return final_score

    # Traceback
    aligned_a, aligned_b = [], []
    i, j = n, m
    while i > 0 or j > 0:
        move = traceback[i, j]
        if move == 0:
            aligned_a.append(seq_a[i - 1])
            aligned_b.append(seq_b[j - 1])
            i -= 1
            j -= 1
        elif move == 1:
            aligned_a.append(seq_a[i - 1])
            aligned_b.append('-')
            i -= 1
        else:
            aligned_a.append('-')
            aligned_b.append(seq_b[j - 1])
            j -= 1

    aligned_a.reverse()
    aligned_b.reverse()
    return final_score, "".join(aligned_a), "".join(aligned_b)


# =============================================================================
# Protein Alignment (Needleman-Wunsch + BLOSUM62)
# =============================================================================

def protein_alignment(protein_a: str, protein_b: str,
                      substitution_matrix: Dict[Tuple[str, str], int],
                      gap_penalty: int = -2,
                      return_alignment: bool = False) -> Union[float, Tuple[float, str, str]]:
    """
    Global protein alignment using Needleman-Wunsch with BLOSUM62.
    """
    protein_a = protein_a.upper()
    protein_b = protein_b.upper()
    n, m = len(protein_a), len(protein_b)

    dp = np.zeros((n + 1, m + 1), dtype=float)
    traceback = np.zeros((n + 1, m + 1), dtype=np.int8)

    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + gap_penalty
        traceback[i, 0] = 1
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + gap_penalty
        traceback[0, j] = 2

    for i in range(1, n + 1):
        aa_a = protein_a[i - 1]
        for j in range(1, m + 1):
            aa_b = protein_b[j - 1]
            sub_score = get_substitution_score(substitution_matrix, aa_a, aa_b)

            diag_score = dp[i - 1, j - 1] + sub_score
            up_score = dp[i - 1, j] + gap_penalty
            left_score = dp[i, j - 1] + gap_penalty

            best_score = diag_score
            best_move = 0
            if up_score > best_score:
                best_score = up_score
                best_move = 1
            if left_score > best_score:
                best_score = left_score
                best_move = 2

            dp[i, j] = best_score
            traceback[i, j] = best_move

    final_score = float(dp[n, m])

    if not return_alignment:
        return final_score

    aligned_a, aligned_b = [], []
    i, j = n, m
    while i > 0 or j > 0:
        move = traceback[i, j]
        if move == 0:
            aligned_a.append(protein_a[i - 1])
            aligned_b.append(protein_b[j - 1])
            i -= 1
            j -= 1
        elif move == 1:
            aligned_a.append(protein_a[i - 1])
            aligned_b.append('-')
            i -= 1
        else:
            aligned_a.append('-')
            aligned_b.append(protein_b[j - 1])
            j -= 1

    aligned_a.reverse()
    aligned_b.reverse()
    return final_score, "".join(aligned_a), "".join(aligned_b)


# =============================================================================
# Codon Alignment (Frameshift-Aware)
# =============================================================================

def codon_alignment(seq_a: str, seq_b: str,
                    substitution_matrix: Dict[Tuple[str, str], int]) -> float:
    """
    Frameshift-aware codon alignment.

    Returns a score (not directly convertible to evolutionary distance).
    Used for qualitative comparison of codon-level similarity.
    """
    seq_a = seq_a.upper().replace('U', 'T')
    seq_b = seq_b.upper().replace('U', 'T')
    len_a, len_b = len(seq_a), len(seq_b)

    dp = np.full((len_a + 1, len_b + 1), float('-inf'))
    dp[0, 0] = 0.0

    # Initialize codon-aligned positions
    for i in range(3, len_a + 1, 3):
        dp[i, 0] = dp[i - 3, 0] + CODON_GAP_PENALTY
    for j in range(3, len_b + 1, 3):
        dp[0, j] = dp[0, j - 3] + CODON_GAP_PENALTY

    for i in range(len_a + 1):
        for j in range(len_b + 1):
            if i == 0 and j == 0:
                continue

            current_best = dp[i, j]

            # Codon match (3 vs 3)
            if i >= 3 and j >= 3:
                codon_a = seq_a[i - 3:i]
                codon_b = seq_b[j - 3:j]
                aa_a = CODON_TABLE.get(codon_a, 'X')
                aa_b = CODON_TABLE.get(codon_b, 'X')
                score = get_substitution_score(substitution_matrix, aa_a, aa_b)
                current_best = max(current_best, dp[i - 3, j - 3] + score)

            # Frameshifts
            if i >= 3 and j >= 2:
                current_best = max(current_best, dp[i - 3, j - 2] + FRAME_SHIFT_PENALTY)
            if i >= 3 and j >= 1:
                current_best = max(current_best, dp[i - 3, j - 1] + FRAME_SHIFT_PENALTY)
            if i >= 2 and j >= 3:
                current_best = max(current_best, dp[i - 2, j - 3] + FRAME_SHIFT_PENALTY)
            if i >= 1 and j >= 3:
                current_best = max(current_best, dp[i - 1, j - 3] + FRAME_SHIFT_PENALTY)

            # Codon gaps
            if i >= 3:
                current_best = max(current_best, dp[i - 3, j] + CODON_GAP_PENALTY)
            if j >= 3:
                current_best = max(current_best, dp[i, j - 3] + CODON_GAP_PENALTY)

            dp[i, j] = current_best

    return float(dp[len_a, len_b])


# =============================================================================
# Test Suite
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SEQUENCE ALIGNMENT ALGORITHMS - VALIDATION")
    print("=" * 70)

    BLOSUM62 = add_stop_penalties(dict(BLOSUM62_NO_STOP), stop_penalty=-5)

    # DNA test
    print("\n[TEST 1] DNA Alignment")
    print("-" * 50)
    seq1, seq2 = "GATTACA", "GCATGCA"
    score, aln1, aln2 = dna_alignment(seq1, seq2, return_alignment=True)
    print(f"  Seq A: {seq1}")
    print(f"  Seq B: {seq2}")
    print(f"  Aligned A: {aln1}")
    print(f"  Aligned B: {aln2}")
    print(f"  Score: {score}")

    # Protein test
    print("\n[TEST 2] Protein Alignment")
    print("-" * 50)
    p1, p2 = "MKTA", "MRTA"
    score, aln1, aln2 = protein_alignment(p1, p2, BLOSUM62, return_alignment=True)
    print(f"  Seq A: {p1}")
    print(f"  Seq B: {p2}")
    print(f"  Aligned A: {aln1}")
    print(f"  Aligned B: {aln2}")
    print(f"  Score: {score}")

    # Codon test
    print("\n[TEST 3] Codon Alignment")
    print("-" * 50)
    d1, d2 = "ATGGAAGCT", "ATGAAAGCT"
    score = codon_alignment(d1, d2, BLOSUM62)
    print(f"  Seq A: {d1} ({translate_dna_to_protein(d1)})")
    print(f"  Seq B: {d2} ({translate_dna_to_protein(d2)})")
    print(f"  Score: {score}")

    # Self-alignment test
    print("\n[TEST 4] Self-Alignment Validation")
    print("-" * 50)
    test_seq = "ATGGCCGATTACAAA"
    self_score = dna_alignment(test_seq, test_seq)
    expected = len(test_seq) * MATCH_SCORE_DNA
    print(f"  Sequence: {test_seq} ({len(test_seq)} bp)")
    print(f"  Self-alignment score: {self_score}")
    print(f"  Expected: {expected}")
    print(f"  Status: {'PASS' if self_score == expected else 'FAIL'}")

    print("\n" + "=" * 70)
    print("All tests completed.")
    print("=" * 70)