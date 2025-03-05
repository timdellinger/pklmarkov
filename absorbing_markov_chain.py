# reusable code courtesy of Claude

import numpy as np
import pandas as pd
import numpy.linalg as la

class AbsorbingMarkovChain:
    def __init__(self, transition_matrix, state_labels=None):
        """
        Initialize the Markov Chain with a transition matrix and optional state labels
        
        Args:
            transition_matrix (np.ndarray or pd.DataFrame): Transition probability matrix
            state_labels (list, optional): List of state names/labels
        """
        # Convert to DataFrame if numpy array
        if isinstance(transition_matrix, np.ndarray):
            # If no labels provided, create generic state labels
            if state_labels is None:
                state_labels = [f'State_{i}' for i in range(transition_matrix.shape[0])]
            
            self.P = pd.DataFrame(
                transition_matrix, 
                index=state_labels, 
                columns=state_labels
            )
        elif isinstance(transition_matrix, pd.DataFrame):
            self.P = transition_matrix
            # If labels not explicitly provided, use DataFrame's existing labels
            state_labels = list(self.P.index)
        else:
            raise ValueError("Input must be numpy array or pandas DataFrame")
        
        # Verify stochastic matrix properties
        assert np.allclose(self.P.values.sum(axis=1), 1), "Rows must sum to 1"
        
        self.state_labels = state_labels
        
    def identify_canonical_form(self):
        """
        Identify absorbing and transient states while preserving labels
        
        Returns:
            dict: Canonical form details with labeled matrices
        """
        # Identify absorbing states (states where transition probability to self is 1)
        absorbing_states = [
            label for label, row in self.P.iterrows() 
            if np.isclose(row.loc[label], 1.0)
        ]
        
        transient_states = [
            label for label in self.P.index 
            if label not in absorbing_states
        ]
        
        # Reorder matrix to put absorbing states last
        full_order = transient_states + absorbing_states
        canonical_P = self.P.loc[full_order, full_order]
        
        # Split canonical matrix into submatrices
        r = len(transient_states)
        Q = canonical_P.iloc[:r, :r]      # Transient-to-transient transitions
        R = canonical_P.iloc[:r, r:]      # Transient-to-absorbing transitions
        
        return {
            'canonical_matrix': canonical_P,
            'Q': Q,
            'R': R,
            'transient_states': transient_states,
            'absorbing_states': absorbing_states
        }
    
    def fundamental_matrix(self):
        """
        Compute fundamental matrix F = (I-Q)^-1 with preserved labels
        
        Returns:
            pd.DataFrame: Labeled fundamental matrix
        """
        canonical_info = self.identify_canonical_form()
        Q = canonical_info['Q']
        
        # Compute fundamental matrix
        I = pd.DataFrame(
            np.eye(Q.shape[0]), 
            index=Q.index, 
            columns=Q.columns
        )
        F = pd.DataFrame(
            la.inv(I.values - Q.values),
            index=Q.index,
            columns=Q.columns
        )
        
        return F
    
    def absorption_probabilities(self):
        """
        Compute absorption probabilities for each transient state
        
        Returns:
            pd.DataFrame: DataFrame with labeled absorption probabilities
        """
        canonical_info = self.identify_canonical_form()
        F = self.fundamental_matrix()
        R = canonical_info['R']
        
        # Compute expected number of steps before absorption
        B = pd.DataFrame(
            F.values @ R.values,
            index=F.index,
            columns=R.columns
        )
        
        return B

"""

# Example usage with custom labels
transition_matrix = np.array([
    [0.7, 0.2, 0.1, 0.0],
    [0.3, 0.5, 0.2, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

# Custom state labels
state_labels = ['Initial', 'Intermediate', 'First_Absorbing', 'Second_Absorbing']

markov_chain = AbsorbingMarkovChain(transition_matrix, state_labels)

# Get canonical form details
canonical_details = markov_chain.identify_canonical_form()
print("Canonical Matrix:")
print(canonical_details['canonical_matrix'])
print("\nTransient States:", canonical_details['transient_states'])
print("Absorbing States:", canonical_details['absorbing_states'])

# Compute fundamental matrix
F = markov_chain.fundamental_matrix()
print("\nFundamental Matrix:")
print(F)

# Absorption Probabilities
absorption_probs = markov_chain.absorption_probabilities()
print("\nAbsorption Probabilities:")
print(absorption_probs)

""";
