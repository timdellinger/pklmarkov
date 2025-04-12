# this is the final version (so far)
# thanks to Claude

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import linalg as spla

class SparseAbsorbingMarkovChain:
    def __init__(self, transition_matrix, state_labels=None):
        """
        Initialize the Markov Chain with a sparse transition matrix and optional state labels

        Args:
            transition_matrix: Can be numpy array, pandas DataFrame, or scipy sparse matrix
            state_labels (list, optional): List of state names/labels
        """
        self.n = None  # Will be set during conversion

        # Handle different input types
        if isinstance(transition_matrix, pd.DataFrame):
            self.state_labels = list(transition_matrix.index)
            # Convert to sparse matrix in CSR format for efficient row operations
            self.P = sp.csr_matrix(transition_matrix.values)
            self.n = len(self.state_labels)

        elif isinstance(transition_matrix, np.ndarray):
            self.n = transition_matrix.shape[0]
            if state_labels is None:
                # self.state_labels = [f'State_{i}' for i in range(self.n)]
                self.state_labels = [int(i) for i in range(self.n)]

            else:
                self.state_labels = state_labels
            # Convert to sparse matrix
            self.P = sp.csr_matrix(transition_matrix)

        elif sp.issparse(transition_matrix):
            # Use CSR format for row operations
            self.P = transition_matrix.tocsr()
            self.n = self.P.shape[0]
            if state_labels is None:
                # self.state_labels = [f'State_{i}' for i in range(self.n)]
                self.state_labels = [int(i) for i in range(self.n)]

            else:
                self.state_labels = state_labels

        else:
            raise ValueError("Input must be numpy array, pandas DataFrame, or scipy sparse matrix")

        # Verify stochastic matrix properties
        row_sums = self.P.sum(axis=1).A1  # Convert to 1D array
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Rows must sum to 1")

    def identify_canonical_form(self):
        """
        Identify absorbing and transient states while preserving labels

        Returns:
            dict: Canonical form components with labels
        """
        # Get diagonal elements to identify absorbing states
        diag_elements = self.P.diagonal()

        # Identify absorbing states (diagonal element = 1)
        absorbing_indices = np.where(np.isclose(diag_elements, 1.0))[0]
        absorbing_states = [self.state_labels[i] for i in absorbing_indices]

        # Identify transient states
        transient_indices = np.setdiff1d(range(self.n), absorbing_indices)
        transient_states = [self.state_labels[i] for i in transient_indices]

        # Full ordering for canonical form
        full_order_indices = list(transient_indices) + list(absorbing_indices)
        full_order_labels = transient_states + absorbing_states

        # Create index mapper for reordering
        state_to_index = {state: i for i, state in enumerate(self.state_labels)}
        canonical_order = [state_to_index[state] for state in full_order_labels]

        # Reorder matrix using sparse indexing
        canonical_P = self.P[canonical_order, :][:, canonical_order]

        # Split canonical matrix into submatrices
        r = len(transient_states)
        Q = canonical_P[:r, :r]      # Transient-to-transient transitions
        R = canonical_P[:r, r:]      # Transient-to-absorbing transitions

        return {
            'canonical_matrix': canonical_P,
            'Q': Q,
            'R': R,
            'transient_states': transient_states,
            'absorbing_states': absorbing_states,
            'transient_indices': transient_indices,
            'absorbing_indices': absorbing_indices
        }

    def fundamental_matrix(self):
        """
        Compute fundamental matrix F = (I-Q)^-1 with preserved labels

        Returns:
            pd.DataFrame: Labeled fundamental matrix
        """
        canonical_info = self.identify_canonical_form()
        Q = canonical_info['Q']
        transient_states = canonical_info['transient_states']

        # Convert to CSC format for efficient solver operations
        Q_csc = Q.tocsc()

        # Create identity matrix in CSC format
        I = sp.eye(Q.shape[0], format='csc')

        # Compute (I-Q) in CSC format
        IQ_diff = I - Q_csc

        # For very large sparse matrices, use iterative solver
        # For smaller matrices, direct solver is more efficient
        if Q.shape[0] > 1000:
            # Using iterative solver for large matrices
            F_values = np.zeros((Q.shape[0], Q.shape[0]))

            # Create sparse identity matrix in CSC format for right-hand side
            I_dense = np.eye(Q.shape[0])

            for i in range(Q.shape[0]):
                b = I_dense[:, i]  # Use column of identity matrix as right-hand side
                x, info = spla.gmres(IQ_diff, b)
                if info != 0:
                    raise ValueError(f"GMRES solver did not converge for column {i}")
                F_values[:, i] = x
        else:
            # Direct solver for smaller matrices - Using CSC format to avoid warnings
            try:
                # Use spsolve with CSC format matrices to eliminate warnings
                F_sparse = spla.spsolve(IQ_diff, I)
                # Convert result to array
                F_values = F_sparse.toarray() if sp.issparse(F_sparse) else F_sparse
            except Exception as e:
                print(f"Sparse solver error: {e}. Falling back to dense computation.")
                # Fallback to dense computation if sparse solver fails
                F_values = np.linalg.inv((I - Q).toarray())

        # Convert to DataFrame with labels
        F = pd.DataFrame(
            F_values,
            index=transient_states,
            columns=transient_states
        )

        return F

    def compute_FA(self):
        """
        Compute the solution F*A (where A is the R matrix in canonical form)

        Returns:
            pd.DataFrame: Labeled absorption probabilities (F*A)
        """
        canonical_info = self.identify_canonical_form()
        F = self.fundamental_matrix()
        R = canonical_info['R']

        # Convert sparse R to dense for matrix multiplication with F
        R_dense = R.toarray()

        # Compute F*A (A is R in our notation)
        FA = pd.DataFrame(
            F.values @ R_dense,
            index=F.index,
            columns=[self.state_labels[i] for i in canonical_info['absorbing_indices']]
        )

        return FA

    def get_labeled_canonical_matrix(self):
        """
        Get canonical form matrix as a pandas DataFrame with proper labels

        Returns:
            pd.DataFrame: Canonical form matrix with labels
        """
        canonical_info = self.identify_canonical_form()
        full_order = canonical_info['transient_states'] + canonical_info['absorbing_states']

        # Create DataFrame from sparse matrix
        return pd.DataFrame(
            canonical_info['canonical_matrix'].toarray(),
            index=full_order,
            columns=full_order
        )
'''
# Example usage with sparse matrix
def create_example_sparse_matrix(n=1000, density=0.003):
    """Create a large sparse transition matrix for testing"""
    # Create random sparse matrix
    random_matrix = sp.random(n, n, density=density, format='csr')

    # Make some states absorbing (diagonal = 1, row = 0 except diagonal)
    absorbing_states = np.random.choice(range(n), size=int(n*0.1), replace=False)

    # Set rows to 0 for absorbing states
    for state in absorbing_states:
        random_matrix[state, :] = 0

    # Set diagonal to 1 for absorbing states
    for state in absorbing_states:
        random_matrix[state, state] = 1

    # Normalize rows of non-absorbing states to sum to 1
    for i in range(n):
        if i not in absorbing_states:
            row_sum = random_matrix[i, :].sum()
            if row_sum > 0:
                random_matrix[i, :] = random_matrix[i, :] / row_sum
            else:
                # If row sum is 0, make it transition to random state
                j = np.random.randint(0, n)
                random_matrix[i, j] = 1.0

    return random_matrix

# Small example for demonstration
small_matrix = np.array([
    [0.7, 0.2, 0.1, 0.0],
    [0.3, 0.5, 0.2, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

# Convert to sparse matrix
small_sparse = sp.csr_matrix(small_matrix)

# Custom state labels
state_labels = ['Initial', 'Intermediate', 'First_Absorbing', 'Second_Absorbing']

markov_chain = SparseAbsorbingMarkovChain(small_sparse, state_labels)

# Get canonical form details
canonical_details = markov_chain.identify_canonical_form()
print("Canonical Matrix:")
print(markov_chain.get_labeled_canonical_matrix())
print("\nTransient States:", canonical_details['transient_states'])
print("Absorbing States:", canonical_details['absorbing_states'])

# Compute fundamental matrix
F = markov_chain.fundamental_matrix()
print("\nFundamental Matrix:")
print(F)

# Compute F*A
FA = markov_chain.compute_FA()
print("\nF*A Matrix (Absorption Probabilities):")
print(FA)

# Example of creating a large sparse matrix
print("\nCreating large sparse matrix example:")
large_sparse = create_example_sparse_matrix(n=1000, density=0.003)
print(f"Matrix shape: {large_sparse.shape}")
print(f"Number of non-zero elements: {large_sparse.nnz}")
print(f"Density: {large_sparse.nnz / (large_sparse.shape[0] * large_sparse.shape[1]):.6f}")

# Process large sparse matrix
large_chain = SparseAbsorbingMarkovChain(large_sparse)
# For large matrices, we might just print a summary of the results
print("\nLarge matrix canonical form summary:")
canonical_large = large_chain.identify_canonical_form()
print(f"Number of transient states: {len(canonical_large['transient_states'])}")
print(f"Number of absorbing states: {len(canonical_large['absorbing_states'])}")
''';
