"""
Phylogenetic Tree Construction using Neighbor-Joining

Builds phylogenetic trees from distance matrices using the
Neighbor-Joining algorithm, which does not assume a molecular clock.


Authors:
  Wassan Haj Yahia
  Salah Mahmied
  Abed Jabaly
  Rawaa Aburaya
  Juman Abu Rmeleh
Course: Computational Biology
"""

import numpy as np
import csv
from typing import List, Tuple, Optional, Dict
from matplotlib import pyplot as plt


# =============================================================================
# File I/O
# =============================================================================

def read_matrix(filepath: str) -> Tuple[List[str], np.ndarray]:
    """Read distance matrix from CSV file."""
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        species = header[1:]
        data = [[float(x) for x in row[1:]] for row in reader]

    matrix = np.array(data, dtype=float)
    if matrix.shape[0] != len(species):
        raise ValueError("Matrix dimensions don't match species count")

    return species, matrix


def validate_matrix(matrix: np.ndarray) -> np.ndarray:
    """Check and fix common matrix issues."""
    if not np.allclose(matrix, matrix.T, rtol=1e-5, equal_nan=True):
        print("  [!] Matrix not symmetric, averaging...")
        matrix = (matrix + matrix.T) / 2

    if not np.allclose(np.diag(matrix), 0, atol=1e-6):
        np.fill_diagonal(matrix, 0)

    if np.any(np.isinf(matrix)):
        max_val = np.max(matrix[np.isfinite(matrix)])
        matrix = np.where(np.isinf(matrix), max_val * 2, matrix)

    return matrix


# =============================================================================
# Neighbor-Joining Algorithm
# =============================================================================

class NeighborJoining:
    """
    Neighbor-Joining tree construction.

    Unlike UPGMA, this algorithm doesn't assume all lineages evolve
    at the same rate, producing more realistic branch lengths.
    """

    def __init__(self, species: List[str], dist_matrix: np.ndarray):
        self.original_species = species.copy()
        self.species = species.copy()
        self.dist = dist_matrix.copy()
        self.tree: Dict[str, Optional[Tuple]] = {s: None for s in species}
        self.branch_lengths: Dict[str, float] = {}
        self._done = False

    def build(self) -> str:
        """Run the algorithm and return root node name."""
        if self._done:
            return self._root

        while len(self.species) > 2:
            Q = self._q_matrix()
            i, j = self._min_q_pair(Q)
            br_i, br_j = self._branch_lengths(i, j)
            new_node = self._join(i, j, br_i, br_j)
            self._update_distances(i, j, new_node)

        self._final_join()
        self._done = True
        return self._root

    def _q_matrix(self) -> np.ndarray:
        """Compute Q-matrix for neighbor selection."""
        n = len(self.species)
        Q = np.zeros((n, n))
        row_sums = np.sum(self.dist, axis=1)

        for i in range(n):
            for j in range(i + 1, n):
                q = (n - 2) * self.dist[i, j] - row_sums[i] - row_sums[j]
                Q[i, j] = Q[j, i] = q

        np.fill_diagonal(Q, np.inf)
        return Q

    def _min_q_pair(self, Q: np.ndarray) -> Tuple[int, int]:
        """Find pair with minimum Q value."""
        idx = np.argmin(Q)
        return np.unravel_index(idx, Q.shape)

    def _branch_lengths(self, i: int, j: int) -> Tuple[float, float]:
        """Calculate branch lengths to new node."""
        n = len(self.species)
        d_ij = self.dist[i, j]

        if n == 2:
            return d_ij / 2, d_ij / 2

        sum_i = np.sum(self.dist[i, :])
        sum_j = np.sum(self.dist[j, :])

        br_i = d_ij / 2 + (sum_i - sum_j) / (2 * (n - 2))
        br_j = d_ij - br_i

        return max(0, br_i), max(0, br_j)

    def _join(self, i: int, j: int, br_i: float, br_j: float) -> str:
        """Create new internal node."""
        node_i, node_j = self.species[i], self.species[j]
        new_node = f"({node_i},{node_j})"

        self.tree[new_node] = (node_i, node_j, br_i, br_j)
        self.branch_lengths[node_i] = br_i
        self.branch_lengths[node_j] = br_j

        return new_node

    def _update_distances(self, i: int, j: int, new_node: str):
        """Update matrix after joining."""
        n = len(self.species)
        new_dists = []

        for k in range(n):
            if k not in (i, j):
                d = (self.dist[i, k] + self.dist[j, k] - self.dist[i, j]) / 2
                new_dists.append((self.species[k], max(0, d)))

        for idx in sorted([i, j], reverse=True):
            self.species.pop(idx)
            self.dist = np.delete(self.dist, idx, axis=0)
            self.dist = np.delete(self.dist, idx, axis=1)

        self.species.append(new_node)
        n_new = len(self.species)
        new_mat = np.zeros((n_new, n_new))
        new_mat[:-1, :-1] = self.dist

        for sp, d in new_dists:
            k = self.species.index(sp)
            new_mat[k, -1] = new_mat[-1, k] = d

        self.dist = new_mat

    def _final_join(self):
        """Join last two nodes."""
        a, b = self.species[0], self.species[1]
        d = self.dist[0, 1]

        self._root = f"({a},{b})"
        self.tree[self._root] = (a, b, d / 2, d / 2)
        self.branch_lengths[a] = d / 2
        self.branch_lengths[b] = d / 2

    def newick(self) -> str:
        """Generate Newick format string."""
        if not self._done:
            self.build()

        def recurse(node):
            if self.tree[node] is None:
                return node
            left, right, br_l, br_r = self.tree[node]
            return f"({recurse(left)}:{br_l:.4f},{recurse(right)}:{br_r:.4f})"

        return recurse(self._root) + ";"

    def plot(self, title: str = "Phylogenetic Tree",
             figsize: Tuple[int, int] = (12, 8),
             save_path: Optional[str] = None,
             show: bool = True) -> plt.Figure:
        """Draw the tree."""
        if not self._done:
            self.build()

        fig, ax = plt.subplots(figsize=figsize)

        leaf_pos = {}
        leaf_idx = [0]

        def assign_y(node):
            if self.tree[node] is None:
                y = leaf_idx[0]
                leaf_idx[0] += 1
                leaf_pos[node] = y
                return y
            left, right, _, _ = self.tree[node]
            return (assign_y(left) + assign_y(right)) / 2

        def get_depths(node, depth=0):
            depths = {node: depth}
            if self.tree[node]:
                left, right, br_l, br_r = self.tree[node]
                depths.update(get_depths(left, depth + br_l))
                depths.update(get_depths(right, depth + br_r))
            return depths

        assign_y(self._root)
        depths = get_depths(self._root)
        max_d = max(depths.values())

        def get_y(node):
            if self.tree[node] is None:
                return leaf_pos[node]
            left, right, _, _ = self.tree[node]
            return (get_y(left) + get_y(right)) / 2

        def draw(node, px=None, py=None):
            x = max_d - depths[node]

            if self.tree[node] is None:
                y = leaf_pos[node]
                ax.plot(x, y, 'ko', ms=8)
                label = node.replace('mRna_', '').replace('protien_', '')
                ax.text(x + 0.3, y, label, va='center', fontsize=10)
            else:
                left, right, _, _ = self.tree[node]
                y = get_y(node)
                draw(left, x, y)
                draw(right, x, y)

            if px is not None:
                ax.plot([px, px, x], [py, y, y], 'b-', lw=1.5)

        draw(self._root)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Evolutionary Distance", fontsize=12)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved: {save_path}")

        if show:
            plt.show()

        return fig


# =============================================================================
# Main Functions
# =============================================================================

def build_tree(matrix_file: str, title: str = None,
               save_path: str = None, show: bool = True) -> NeighborJoining:
    """Build and display tree from distance matrix file."""

    print(f"\n  Loading: {matrix_file}")
    species, matrix = read_matrix(matrix_file)
    matrix = validate_matrix(matrix)

    print(f"  Species: {len(species)}")

    nj = NeighborJoining(species, matrix)
    nj.build()

    if title is None:
        title = f"Phylogenetic Tree\n{matrix_file}"

    nj.plot(title=title, save_path=save_path, show=show)

    return nj


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHYLOGENETIC TREE CONSTRUCTION")
    print("Neighbor-Joining Algorithm")
    print("=" * 70)

    matrices = [
        ("dna_distance_matrix.csv", "DNA-Based Phylogenetic Tree"),
        ("codon_distance_matrix.csv", "Codon-Based Phylogenetic Tree"),
        ("protein_distance_matrix.csv", "Protein-Based Phylogenetic Tree")
    ]

    print("\nAlgorithm: Neighbor-Joining")
    print("  - Does not assume molecular clock")
    print("  - Produces realistic branch lengths")
    print("  - Standard method in phylogenetics")

    for matrix_file, description in matrices:
        print("\n" + "-" * 70)
        print(f"Building: {description}")

        try:
            tree = build_tree(
                matrix_file,
                title=f"{description}\n(Neighbor-Joining)",
                save_path=matrix_file.replace('.csv', '_tree.png'),
                show=True
            )

            print(f"\n  Newick: {tree.newick()}")

        except FileNotFoundError:
            print(f"  [!] File not found: {matrix_file}")
            print("      Run evolutionary_distance.py first.")
        except Exception as e:
            print(f"  [!] Error: {e}")

    print("\n" + "=" * 70)
    print("TREE CONSTRUCTION COMPLETE")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  - dna_distance_matrix_tree.png")
    print("  - codon_distance_matrix_tree.png")
    print("  - protein_distance_matrix_tree.png")
    print("\nInterpretation:")
    print("  - Branch length = evolutionary distance")
    print("  - Closer nodes = more recently diverged")
    print("  - Trees may differ due to:")
    print("    * Silent mutations in DNA")
    print("    * Selection pressure on proteins")
    print("    * Variable evolutionary rates")
    print("=" * 70 + "\n")