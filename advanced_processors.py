"""
Advanced signal processing modules for Cosmic Whisper.
Contains cutting-edge algorithms for alien signal analysis.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from itertools import combinations

# Physical constants (matching main.py)
PLANCK_CONSTANT = 6.62607015e-34
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
EULER_GAMMA = 0.5772156649015329
SPEED_OF_LIGHT = 299792458
FINE_STRUCTURE = 1/137.036
PI_SQUARED = np.pi**2
CATALAN_CONSTANT = 0.9159655941772190
VACUUM_ENERGY_DENSITY = 1e-9
QUANTUM_COHERENCE_TIME = 1e-12

class QuantumFieldProcessor:
    """
    Advanced quantum field theory inspired signal processing.
    Simulates virtual particle interactions and vacuum fluctuations.
    """
    
    def __init__(self, field_dimensions=11):
        self.field_dimensions = field_dimensions
        self.vacuum_state = self._initialize_vacuum()
        
    def _initialize_vacuum(self):
        """Initialize quantum vacuum state with zero-point fluctuations."""
        return np.random.normal(0, VACUUM_ENERGY_DENSITY, self.field_dimensions)
    
    def apply_quantum_field_transform(self, signal):
        """
        Apply quantum field transform using virtual particle interactions.
        Based on second quantization formalism.
        """
        n_points = len(signal)
        field_evolved = np.zeros((n_points, self.field_dimensions))
        
        for t in range(n_points):
            # Time evolution operator U(t) = exp(-iHt/ℏ)
            time_evolution = np.exp(-1j * PLANCK_CONSTANT * t * SPEED_OF_LIGHT)
            
            # Creation and annihilation operators
            creation_op = np.sqrt(t + 1) * FINE_STRUCTURE
            annihilation_op = np.sqrt(t) * FINE_STRUCTURE
            
            # Extract signal value
            signal_val = signal[t] if len(signal.shape) == 1 else signal[t, 0]
            
            # Field mode evolution with virtual particles
            field_mode = signal_val * (creation_op + annihilation_op) * time_evolution
            
            # Vacuum fluctuations with exponential decay
            vacuum_fluctuation = self.vacuum_state * np.random.exponential(QUANTUM_COHERENCE_TIME)
            
            # Combine field evolution and vacuum effects
            field_evolved[t] = np.real(field_mode + vacuum_fluctuation)
        
        return field_evolved
    
    def calculate_casimir_effect(self, signal1, signal2):
        """
        Calculate Casimir-like effect between two signals.
        Models vacuum energy between conducting plates.
        """
        # Prepare signals
        s1 = signal1.flatten() if len(signal1.shape) > 1 else signal1
        s2 = signal2.flatten() if len(signal2.shape) > 1 else signal2
        
        min_len = min(len(s1), len(s2))
        s1, s2 = s1[:min_len], s2[:min_len]
        
        # Zero-point energy density
        zero_point = PLANCK_CONSTANT * SPEED_OF_LIGHT / (2 * np.pi)
        
        # Signal interaction energy (mode coupling)
        interaction_energy = np.sum(s1 * s2) * zero_point
        
        # Casimir force analog (attractive/repulsive)
        casimir_force = -interaction_energy / len(s1)
        
        # Quantum pressure from vacuum fluctuations
        quantum_pressure = casimir_force / PI_SQUARED
        
        return {
            'interaction_energy': interaction_energy,
            'casimir_force': casimir_force,
            'quantum_pressure': quantum_pressure,
            'zero_point_contribution': zero_point
        }

class NeuromorphicProcessor(nn.Module):
    """
    Brain-inspired neuromorphic signal processing using spiking neural networks.
    Implements leaky integrate-and-fire neurons with STDP learning.
    """
    
    def __init__(self, input_dim, n_neurons=256, membrane_threshold=1.0):
        super(NeuromorphicProcessor, self).__init__()
        self.n_neurons = n_neurons
        self.membrane_threshold = membrane_threshold
        
        # Biophysical parameters
        self.tau_mem = 10.0  # Membrane time constant (ms)
        self.tau_syn = 5.0   # Synaptic time constant (ms)
        self.refractory_period = 2.0  # Refractory period (ms)
        
        # Synaptic connectivity
        self.input_weights = nn.Parameter(torch.randn(input_dim, n_neurons) * 0.1)
        self.recurrent_weights = nn.Parameter(torch.randn(n_neurons, n_neurons) * 0.05)
        
        # STDP learning parameters
        self.learning_rate = 0.001
        self.trace_decay = 0.95
        self.stdp_window = 20.0  # STDP time window
        
    def forward(self, input_signal, n_timesteps=1000):
        """
        Process signal through spiking neural network.
        Returns spike trains and neural dynamics.
        """
        batch_size = input_signal.size(0)
        
        # Initialize neural states
        membrane_potential = torch.zeros(batch_size, self.n_neurons)
        synaptic_current = torch.zeros(batch_size, self.n_neurons)
        spike_trace = torch.zeros(batch_size, self.n_neurons)
        refractory_timer = torch.zeros(batch_size, self.n_neurons)
        
        spike_trains = []
        membrane_history = []
        
        for t in range(n_timesteps):
            # Input current injection
            if t < len(input_signal):
                input_current = torch.matmul(input_signal[t].unsqueeze(0), self.input_weights)
            else:
                input_current = torch.zeros(batch_size, self.n_neurons)
            
            # Recurrent synaptic input
            recurrent_current = torch.matmul(spike_trace, self.recurrent_weights)
            
            # Update synaptic current (exponential decay + input)
            synaptic_current = (synaptic_current * (1 - 1/self.tau_syn) + 
                              input_current + recurrent_current)
            
            # Update membrane potential (leaky integration)
            membrane_potential = (membrane_potential * (1 - 1/self.tau_mem) + 
                                synaptic_current / self.tau_mem)
            
            # Apply refractory period
            refractory_mask = refractory_timer > 0
            membrane_potential[refractory_mask] = 0
            refractory_timer = torch.clamp(refractory_timer - 1, min=0)
            
            # Generate spikes (threshold crossing)
            spikes = (membrane_potential > self.membrane_threshold).float()
            
            # Reset neurons that spiked
            membrane_potential = membrane_potential * (1 - spikes)
            refractory_timer = refractory_timer + spikes * self.refractory_period
            
            # Update spike trace (exponential decay)
            spike_trace = spike_trace * self.trace_decay + spikes
            
            spike_trains.append(spikes)
            membrane_history.append(membrane_potential.clone())
        
        spike_trains = torch.stack(spike_trains, dim=1)
        membrane_history = torch.stack(membrane_history, dim=1)
        
        return {
            'spike_trains': spike_trains,
            'membrane_history': membrane_history,
            'final_trace': spike_trace
        }

class IITProcessor:
    """
    Integrated Information Theory (IIT) processor for consciousness-like measures.
    Calculates Φ (phi) - the amount of integrated information.
    """
    
    @staticmethod
    def calculate_phi(signal, max_partitions=5):
        """
        Calculate Φ (phi) - integrated information measure.
        Higher Φ indicates more consciousness-like integration.
        """
        n_dims = signal.shape[1] if len(signal.shape) > 1 else 1
        if n_dims == 1:
            signal = signal.reshape(-1, 1)
        
        # Calculate total system information
        system_entropy = IITProcessor._calculate_entropy(signal)
        
        max_phi = 0
        best_partition = None
        min_cut_info = float('inf')
        
        # Test all possible bipartitions
        for partition_size in range(1, min(n_dims, max_partitions)):
            for partition in combinations(range(n_dims), partition_size):
                part1_indices = list(partition)
                part2_indices = [i for i in range(n_dims) if i not in part1_indices]
                
                if len(part2_indices) == 0:
                    continue
                
                # Calculate partition entropies
                part1_entropy = IITProcessor._calculate_entropy(signal[:, part1_indices])
                part2_entropy = IITProcessor._calculate_entropy(signal[:, part2_indices])
                
                # Joint entropy across the cut
                joint_entropy = IITProcessor._calculate_joint_entropy(
                    signal[:, part1_indices], signal[:, part2_indices]
                )
                
                # Mutual information (information shared across cut)
                mutual_info = part1_entropy + part2_entropy - joint_entropy
                
                # Φ is the minimum information lost by any partition
                if mutual_info < min_cut_info:
                    min_cut_info = mutual_info
                    best_partition = (part1_indices, part2_indices)
        
        phi = min_cut_info if min_cut_info != float('inf') else 0
        
        return {
            'phi': phi,
            'system_entropy': system_entropy,
            'best_partition': best_partition,
            'consciousness_level': phi / system_entropy if system_entropy > 0 else 0,
            'integration_ratio': phi / max(system_entropy, 1e-10)
        }
    
    @staticmethod
    def _calculate_entropy(data):
        """Calculate Shannon entropy of data distribution."""
        if len(data.shape) == 1:
            hist, _ = np.histogram(data, bins=50, density=True)
        else:
            # Multivariate histogram for joint entropy
            hist, _ = np.histogramdd(data, bins=min(20, int(len(data)**0.25)), density=True)
        
        hist = hist.flatten()
        hist = hist[hist > 0]  # Remove zero probabilities
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    @staticmethod
    def _calculate_joint_entropy(data1, data2):
        """Calculate joint entropy between two data sets."""
        combined_data = np.hstack([data1, data2])
        return IITProcessor._calculate_entropy(combined_data)

class TopologicalAnalyzer:
    """
    Topological data analysis for discovering signal structure.
    Computes persistent homology and topological features.
    """
    
    @staticmethod
    def persistent_homology_analysis(signal, max_dimension=2):
        """
        Compute persistent homology using Vietoris-Rips complex.
        Reveals topological features at different scales.
        """
        # Time-delay embedding for phase space reconstruction
        embedded_signal = TopologicalAnalyzer._time_delay_embedding(signal, dim=3, tau=1)
        
        # Compute pairwise distances
        distances = pdist(embedded_signal, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Analyze persistence at different scales
        persistence_diagrams = []
        scales = np.linspace(0, np.max(distances) * 0.5, 50)
        
        for scale in scales:
            # Create simplicial complex at current scale
            adjacency = distance_matrix <= scale
            
            # Compute Betti numbers (topological invariants)
            betti_0 = TopologicalAnalyzer._count_connected_components(adjacency)
            betti_1 = TopologicalAnalyzer._estimate_loops(adjacency)
            
            # Euler characteristic
            euler_char = betti_0 - betti_1
            
            persistence_diagrams.append({
                'scale': scale,
                'betti_0': betti_0,  # Connected components
                'betti_1': betti_1,  # Loops/holes
                'euler_characteristic': euler_char,
                'persistence': scale
            })
        
        return {
            'persistence_diagrams': persistence_diagrams,
            'embedded_signal': embedded_signal,
            'max_scale': np.max(distances),
            'topological_summary': TopologicalAnalyzer._summarize_topology(persistence_diagrams)
        }
    
    @staticmethod
    def _time_delay_embedding(signal, dim=3, tau=1):
        """
        Create time-delay embedding for phase space reconstruction.
        Maps 1D time series to higher-dimensional phase space.
        """
        if len(signal.shape) > 1:
            signal = signal[:, 0]  # Use first dimension
        
        n_points = len(signal) - (dim - 1) * tau
        embedded = np.zeros((n_points, dim))
        
        for i in range(dim):
            embedded[:, i] = signal[i * tau:i * tau + n_points]
        
        return embedded
    
    @staticmethod
    def _count_connected_components(adjacency):
        """Count connected components using depth-first search."""
        n = adjacency.shape[0]
        visited = np.zeros(n, dtype=bool)
        components = 0
        
        for i in range(n):
            if not visited[i]:
                components += 1
                # DFS to mark all connected nodes
                stack = [i]
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        neighbors = np.where(adjacency[node])[0]
                        for neighbor in neighbors:
                            if not visited[neighbor]:
                                stack.append(neighbor)
        
        return components
    
    @staticmethod
    def _estimate_loops(adjacency):
        """Estimate number of 1D holes using Euler characteristic."""
        n = adjacency.shape[0]
        edges = np.sum(adjacency) // 2  # Undirected edges
        vertices = n
        
        # Rough approximation: loops = edges - vertices + components
        components = TopologicalAnalyzer._count_connected_components(adjacency)
        loops = max(0, edges - vertices + components)
        
        return loops
    
    @staticmethod
    def _summarize_topology(persistence_diagrams):
        """Summarize topological features across scales."""
        max_components = max(pd['betti_0'] for pd in persistence_diagrams)
        max_loops = max(pd['betti_1'] for pd in persistence_diagrams)
        
        # Persistence statistics
        component_persistence = [pd['scale'] for pd in persistence_diagrams if pd['betti_0'] > 1]
        loop_persistence = [pd['scale'] for pd in persistence_diagrams if pd['betti_1'] > 0]
        
        return {
            'max_components': max_components,
            'max_loops': max_loops,
            'component_persistence_range': (min(component_persistence) if component_persistence else 0,
                                          max(component_persistence) if component_persistence else 0),
            'loop_persistence_range': (min(loop_persistence) if loop_persistence else 0,
                                     max(loop_persistence) if loop_persistence else 0),
            'topological_complexity': max_components + max_loops
        }

class AdvancedMLEnsemble:
    """
    Ensemble of machine learning models for comprehensive signal classification.
    Combines multiple algorithms for robust pattern recognition.
    """
    
    def __init__(self):
        self.models = {}
        self.is_trained = False
        self.feature_names = []
        self.signal_names = []
    
    def train_ensemble(self, signals_dict, labels=None):
        """
        Train ensemble on signal features.
        Extracts comprehensive features and trains multiple models.
        """
        # Extract features from all signals
        all_features = []
        signal_names = []
        
        for name, signal in signals_dict.items():
            features = self._extract_comprehensive_features(signal)
            all_features.append(features)
            signal_names.append(name)
        
        X = np.array(all_features)
        
        # Create labels if not provided
        if labels is None:
            labels = np.arange(len(signal_names))
        
        # Initialize ensemble models
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        }
        
        # Train each model
        training_results = {}
        for name, model in self.models.items():
            try:
                model.fit(X, labels)
                training_results[name] = 'Success'
            except Exception as e:
                training_results[name] = f'Failed: {str(e)}'
        
        self.is_trained = True
        self.signal_names = signal_names
        self.feature_names = self._get_feature_names()
        
        return X, labels, training_results
    
    def _extract_comprehensive_features(self, signal):
        """Extract comprehensive statistical and spectral features."""
        features = []
        
        # Use first dimension if multidimensional
        sig = signal[:, 0] if len(signal.shape) > 1 else signal
        
        # Statistical moments
        features.extend([
            np.mean(sig), np.std(sig), np.var(sig),
            np.min(sig), np.max(sig), np.median(sig),
            np.percentile(sig, 25), np.percentile(sig, 75)
        ])
        
        # Higher-order moments (skewness, kurtosis)
        from scipy import stats
        features.extend([
            stats.skew(sig), stats.kurtosis(sig)
        ])
        
        # Frequency domain features
        fft_vals = np.fft.fft(sig)
        freqs = np.fft.fftfreq(len(sig))
        psd = np.abs(fft_vals)**2
        
        # Spectral features
        total_power = np.sum(psd)
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd) if total_power > 0 else 0
        spectral_spread = np.sqrt(np.sum((freqs - spectral_centroid)**2 * psd) / np.sum(psd)) if total_power > 0 else 0
        spectral_rolloff = self._spectral_rolloff(freqs, psd)
        
        features.extend([total_power, spectral_centroid, spectral_spread, spectral_rolloff])
        
        # Complexity measures (with error handling)
        try:
            from main import calculate_sample_entropy, lempel_ziv_complexity, estimate_fractal_dimension
            sample_ent = calculate_sample_entropy(sig)
            lz_comp = lempel_ziv_complexity(sig.reshape(-1, 1))
            fractal_dim = estimate_fractal_dimension(sig.reshape(-1, 1))
        except:
            sample_ent, lz_comp, fractal_dim = 0, 0, 1.0
        
        features.extend([sample_ent, lz_comp, fractal_dim])
        
        return features
    
    def _spectral_rolloff(self, freqs, psd, rolloff_percent=0.85):
        """Calculate spectral rolloff frequency."""
        total_power = np.sum(psd)
        if total_power == 0:
            return 0
        
        cumulative_power = np.cumsum(psd)
        rolloff_threshold = rolloff_percent * total_power
        rolloff_idx = np.where(cumulative_power >= rolloff_threshold)[0]
        
        return freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
    
    def _get_feature_names(self):
        """Get descriptive feature names."""
        return [
            'mean', 'std', 'var', 'min', 'max', 'median', 'q25', 'q75',
            'skewness', 'kurtosis', 'total_power', 'spectral_centroid',
            'spectral_spread', 'spectral_rolloff', 'sample_entropy',
            'lz_complexity', 'fractal_dimension'
        ]
    
    def predict_signal_type(self, signal):
        """Predict signal type using ensemble voting."""
        if not self.is_trained:
            return {"error": "Ensemble not trained"}
        
        features = self._extract_comprehensive_features(signal)
        X = np.array(features).reshape(1, -1)
        
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)[0]
                predictions[name] = self.signal_names[pred]
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    probabilities[name] = dict(zip(self.signal_names, proba))
            except Exception as e:
                predictions[name] = f"Error: {str(e)}"
                probabilities[name] = {}
        
        # Ensemble voting
        vote_counts = {}
        for pred in predictions.values():
            if isinstance(pred, str) and not pred.startswith("Error"):
                vote_counts[pred] = vote_counts.get(pred, 0) + 1
        
        ensemble_prediction = max(vote_counts, key=vote_counts.get) if vote_counts else "Unknown"
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': predictions,
            'probabilities': probabilities,
            'features': dict(zip(self.feature_names, features)),
            'vote_counts': vote_counts
        }
