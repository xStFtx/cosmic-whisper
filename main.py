import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
try:
    import quaternion
    QUATERNION_AVAILABLE = True
except ImportError:
    print("Warning: quaternion package not available. Using alternative quaternion implementation.")
    QUATERNION_AVAILABLE = False
from scipy.spatial.transform import Rotation as R
from scipy import signal as scipy_signal
from scipy.stats import entropy
from scipy.signal import hilbert, stft, cwt, morlet
from scipy.optimize import minimize
import matplotlib.pyplot as plt
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: plotly not available. 4D visualizations will be skipped.")
    PLOTLY_AVAILABLE = False
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE, UMAP
import warnings
warnings.filterwarnings('ignore')

# Advanced mathematical constants for quantum-inspired processing
PLANCK_CONSTANT = 6.62607015e-34
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
EULER_GAMMA = 0.5772156649015329

# Alternative quaternion implementation if package not available
class SimpleQuaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    @classmethod
    def from_rotation_vector(cls, vec):
        """Create quaternion from rotation vector"""
        if len(vec) != 3:
            raise ValueError("Rotation vector must be 3D")
        
        angle = np.linalg.norm(vec)
        if angle == 0:
            return cls(1, 0, 0, 0)
        
        axis = vec / angle
        half_angle = angle / 2
        sin_half = np.sin(half_angle)
        
        return cls(
            np.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half
        )

def get_signal():
    """
    Generate a sample 3D signal that could represent cosmic data.
    Returns a time series of 3D vectors.
    """
    t = np.linspace(0, 4*np.pi, 1000)
    
    # Create a complex 3D signal with multiple frequency components
    # This could represent cosmic phenomena like pulsar signals or gravitational waves
    signal_x = np.sin(t) + 0.5 * np.sin(3*t) + 0.2 * np.random.randn(len(t))
    signal_y = np.cos(2*t) + 0.3 * np.cos(5*t) + 0.2 * np.random.randn(len(t))
    signal_z = np.sin(0.5*t) * np.cos(4*t) + 0.2 * np.random.randn(len(t))
    
    signal = np.column_stack([signal_x, signal_y, signal_z])
    return signal

def transform_signal(signal):
    """
    Transform the signal to a quaternion representation.
    """
    # Assuming signal is a 3D vector
    if QUATERNION_AVAILABLE:
        q = quaternion.from_rotation_vector(signal)
    else:
        q = SimpleQuaternion.from_rotation_vector(signal)
    return q

def quaternion_over_fourier(quaternion_signal):
    """
    Perform Fourier analysis on quaternion data.
    This function analyzes the frequency components of quaternion signals.
    """
    # Convert quaternions to components for analysis
    w_component = np.array([q.w for q in quaternion_signal])
    x_component = np.array([q.x for q in quaternion_signal])
    y_component = np.array([q.y for q in quaternion_signal])
    z_component = np.array([q.z for q in quaternion_signal])
    
    # Perform FFT on each component
    fft_w = np.fft.fft(w_component)
    fft_x = np.fft.fft(x_component)
    fft_y = np.fft.fft(y_component)
    fft_z = np.fft.fft(z_component)
    
    # Calculate frequencies
    n = len(quaternion_signal)
    frequencies = np.fft.fftfreq(n)
    
    # Calculate magnitude spectrum for each component
    magnitude_w = np.abs(fft_w)
    magnitude_x = np.abs(fft_x)
    magnitude_y = np.abs(fft_y)
    magnitude_z = np.abs(fft_z)
    
    return {
        'frequencies': frequencies,
        'fft_components': {
            'w': fft_w, 'x': fft_x, 'y': fft_y, 'z': fft_z
        },
        'magnitudes': {
            'w': magnitude_w, 'x': magnitude_x, 'y': magnitude_y, 'z': magnitude_z
        }
    }

def plot_results(signal, quaternion_signal, fourier_results):
    """
    Plot the original signal, quaternion representation, and Fourier analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot original 3D signal
    axes[0, 0].plot(signal[:, 0], label='X')
    axes[0, 0].plot(signal[:, 1], label='Y')
    axes[0, 0].plot(signal[:, 2], label='Z')
    axes[0, 0].set_title('Original 3D Signal')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot quaternion components
    w_vals = [q.w for q in quaternion_signal]
    x_vals = [q.x for q in quaternion_signal]
    y_vals = [q.y for q in quaternion_signal]
    z_vals = [q.z for q in quaternion_signal]
    
    axes[0, 1].plot(w_vals, label='w')
    axes[0, 1].plot(x_vals, label='x')
    axes[0, 1].plot(y_vals, label='y')
    axes[0, 1].plot(z_vals, label='z')
    axes[0, 1].set_title('Quaternion Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot frequency spectrum (magnitude)
    freqs = fourier_results['frequencies']
    mags = fourier_results['magnitudes']
    
    # Only plot positive frequencies for clarity
    pos_freq_mask = freqs >= 0
    pos_freqs = freqs[pos_freq_mask]
    
    axes[1, 0].semilogy(pos_freqs, mags['w'][pos_freq_mask], label='w')
    axes[1, 0].semilogy(pos_freqs, mags['x'][pos_freq_mask], label='x')
    axes[1, 0].semilogy(pos_freqs, mags['y'][pos_freq_mask], label='y')
    axes[1, 0].semilogy(pos_freqs, mags['z'][pos_freq_mask], label='z')
    axes[1, 0].set_title('Frequency Spectrum (Magnitude)')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot combined magnitude spectrum
    combined_magnitude = np.sqrt(mags['w']**2 + mags['x']**2 + mags['y']**2 + mags['z']**2)
    axes[1, 1].semilogy(pos_freqs, combined_magnitude[pos_freq_mask])
    axes[1, 1].set_title('Combined Quaternion Magnitude Spectrum')
    axes[1, 1].set_xlabel('Frequency')
    axes[1, 1].set_ylabel('Combined Magnitude')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def generate_fractal_signal(n_points=1000, dimensions=3):
    """
    Generate alien signals based on fractal patterns (Mandelbrot-inspired).
    """
    t = np.linspace(-2, 2, n_points)
    fractal_signal = np.zeros((n_points, dimensions))
    
    for dim in range(dimensions):
        # Create complex plane for each dimension with slight variations
        real = t + dim * 0.1
        imag = np.linspace(-1.5, 1.5, n_points) + dim * 0.05
        c = real[:, np.newaxis] + 1j * imag[np.newaxis, :]
        
        # Mandelbrot-like iteration
        z = np.zeros_like(c)
        for i in range(50):
            mask = np.abs(z) <= 2
            z[mask] = z[mask]**2 + c[mask]
        
        # Extract signal from fractal boundary
        fractal_signal[:, dim] = np.log(np.abs(z).mean(axis=1) + 1)
    
    return fractal_signal

def generate_prime_sequence_signal(n_points=1000, dimensions=3):
    """
    Generate signals based on prime number sequences.
    """
    def sieve_of_eratosthenes(limit):
        primes = []
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        
        for i in range(2, limit + 1):
            if is_prime[i]:
                primes.append(i)
        return primes
    
    primes = sieve_of_eratosthenes(n_points * 10)
    signal = np.zeros((n_points, dimensions))
    
    for dim in range(dimensions):
        # Use different prime-based transformations for each dimension
        if dim == 0:
            # Prime gaps
            gaps = np.diff(primes[:n_points])
            signal[:len(gaps), dim] = np.sin(gaps / 10.0)
        elif dim == 1:
            # Prime density
            density = [len([p for p in primes if p <= i*10]) for i in range(1, n_points+1)]
            signal[:, dim] = np.cos(np.array(density) / 50.0)
        else:
            # Prime oscillations
            prime_osc = [np.sin(p / 100.0) for p in primes[:n_points]]
            signal[:len(prime_osc), dim] = prime_osc
    
    return signal

def generate_chaotic_signal(n_points=1000, dimensions=3):
    """
    Generate signals from chaotic systems (Lorenz attractor-inspired).
    """
    # Lorenz system parameters
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    dt = 0.01
    
    # Initialize
    xyz = np.zeros((n_points, 3))
    xyz[0] = [1.0, 1.0, 1.0]
    
    # Generate Lorenz attractor
    for i in range(1, n_points):
        x, y, z = xyz[i-1]
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        
        xyz[i] = xyz[i-1] + dt * np.array([dxdt, dydt, dzdt])
    
    if dimensions > 3:
        # Add more chaotic dimensions
        extra_dims = dimensions - 3
        extra_signals = np.zeros((n_points, extra_dims))
        for dim in range(extra_dims):
            # Modified chaotic system for additional dimensions
            extra_signals[:, dim] = np.sin(xyz[:, 0] * (dim + 1)) * np.cos(xyz[:, 1] * (dim + 2))
        
        xyz = np.hstack([xyz, extra_signals])
    
    return xyz[:, :dimensions]

# AI-based decoding classes
class SignalAutoencoder(nn.Module):
    """
    Autoencoder for signal compression and feature extraction.
    """
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super(SignalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def decode_signal_with_ai(signal):
    """
    Use AI techniques to decode and analyze the signal.
    """
    # Prepare data
    scaler = StandardScaler()
    signal_normalized = scaler.fit_transform(signal)
    
    # Convert to tensor
    signal_tensor = torch.FloatTensor(signal_normalized)
    
    # Train autoencoder
    input_dim = signal.shape[1]
    autoencoder = SignalAutoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(100):
        optimizer.zero_grad()
        reconstructed, encoded = autoencoder(signal_tensor)
        loss = criterion(reconstructed, signal_tensor)
        loss.backward()
        optimizer.step()
    
    # Extract features
    with torch.no_grad():
        _, latent_features = autoencoder(signal_tensor)
        latent_features = latent_features.numpy()
    
    # Clustering analysis
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_features)
    
    # DBSCAN for anomaly detection
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    anomaly_labels = dbscan.fit_predict(latent_features)
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(latent_features)
    
    return {
        'latent_features': latent_features,
        'cluster_labels': cluster_labels,
        'anomaly_labels': anomaly_labels,
        'pca_features': pca_features,
        'reconstruction_loss': loss.item()
    }

def analyze_signal_complexity(signal):
    """
    Analyze signal complexity using entropy and other measures.
    """
    complexity_metrics = {}
    
    for dim in range(signal.shape[1]):
        dim_signal = signal[:, dim]
        
        # Shannon entropy
        hist, _ = np.histogram(dim_signal, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zeros
        shannon_entropy = entropy(hist)        # Spectral entropy
        f, psd = scipy_signal.periodogram(dim_signal)
        psd_normalized = psd / np.sum(psd)
        psd_normalized = psd_normalized[psd_normalized > 0]
        spectral_entropy = entropy(psd_normalized)
        
        # Approximate entropy
        def approximate_entropy(data, m=2, r=0.2):
            N = len(data)
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([data[i:i+m] for i in range(N-m+1)])
                C = np.zeros(N-m+1)
                for i in range(N-m+1):
                    template = patterns[i]
                    C[i] = sum([1 for j in range(N-m+1) if _maxdist(template, patterns[j], m) <= r])
                    C[i] = C[i] / (N-m+1)
                phi = sum([np.log(c) for c in C if c > 0]) / (N-m+1)
                return phi
            
            return _phi(m) - _phi(m+1)
        
        approx_entropy = approximate_entropy(dim_signal)
        
        complexity_metrics[f'dim_{dim}'] = {
            'shannon_entropy': shannon_entropy,
            'spectral_entropy': spectral_entropy,
            'approximate_entropy': approx_entropy
        }
    
    return complexity_metrics

def create_4d_visualization(signal, quaternion_signal, ai_results):
    """
    Create 4D visualization using quaternions and time.
    """
    if not PLOTLY_AVAILABLE:
        print("📊 Plotly not available - creating alternative matplotlib visualization...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 3D Signal trajectory (projected to 2D)
        time = np.linspace(0, 1, len(signal))
        scatter = axes[0, 0].scatter(signal[:, 0], signal[:, 1], c=time, cmap='viridis', s=10)
        axes[0, 0].set_title('Signal Trajectory (X-Y Projection)')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        plt.colorbar(scatter, ax=axes[0, 0], label='Time')
        
        # Quaternion components
        q_w = [q.w for q in quaternion_signal]
        q_x = [q.x for q in quaternion_signal]
        q_y = [q.y for q in quaternion_signal]
        q_z = [q.z for q in quaternion_signal]
        
        axes[0, 1].plot(q_w, label='w', alpha=0.7)
        axes[0, 1].plot(q_x, label='x', alpha=0.7)
        axes[0, 1].plot(q_y, label='y', alpha=0.7)
        axes[0, 1].plot(q_z, label='z', alpha=0.7)
        axes[0, 1].set_title('Quaternion Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AI latent space (2D projection)
        if ai_results['pca_features'].shape[1] >= 2:
            scatter2 = axes[1, 0].scatter(
                ai_results['pca_features'][:, 0], 
                ai_results['pca_features'][:, 1], 
                c=ai_results['cluster_labels'], 
                cmap='Set1', 
                s=10
            )
            axes[1, 0].set_title('AI Latent Space (PCA)')
            axes[1, 0].set_xlabel('PC1')
            axes[1, 0].set_ylabel('PC2')
            plt.colorbar(scatter2, ax=axes[1, 0], label='Cluster')
        
        # Complexity over time
        complexity_time = np.cumsum(np.abs(np.diff(signal, axis=0)).mean(axis=1))
        axes[1, 1].plot(time[1:], complexity_time, 'b-', linewidth=2)
        axes[1, 1].set_title('Signal Complexity Over Time')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Cumulative Complexity')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        return fig
    
    # Create time array
    time = np.linspace(0, 1, len(signal))
    
    # Create 4D plot with Plotly
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['3D Signal Trajectory', 'Quaternion Space', 'AI Latent Space', 'Complexity Over Time'],
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
               [{'type': 'scatter3d'}, {'type': 'xy'}]]
    )
    
    # 3D Signal trajectory with time as color
    fig.add_trace(
        go.Scatter3d(
            x=signal[:, 0],
            y=signal[:, 1],
            z=signal[:, 2],
            mode='markers+lines',
            marker=dict(
                size=3,
                color=time,
                colorscale='Viridis',
                colorbar=dict(title="Time")
            ),
            name='Signal Trajectory'
        ),
        row=1, col=1
    )
    
    # Quaternion space visualization
    q_w = [q.w for q in quaternion_signal]
    q_x = [q.x for q in quaternion_signal]
    q_y = [q.y for q in quaternion_signal]
    q_z = [q.z for q in quaternion_signal]
    
    fig.add_trace(
        go.Scatter3d(
            x=q_x,
            y=q_y,
            z=q_z,
            mode='markers',
            marker=dict(
                size=3,
                color=q_w,
                colorscale='Plasma',
                colorbar=dict(title="Quaternion W")
            ),
            name='Quaternion Space'
        ),
        row=1, col=2
    )
    
    # AI latent space
    if ai_results['pca_features'].shape[1] >= 3:
        fig.add_trace(
            go.Scatter3d(
                x=ai_results['pca_features'][:, 0],
                y=ai_results['pca_features'][:, 1],
                z=ai_results['pca_features'][:, 2],
                mode='markers',                marker=dict(
                    size=3,
                    color=ai_results['cluster_labels'],
                    colorscale='viridis',
                    colorbar=dict(title="Cluster")
                ),
                name='AI Latent Space'
            ),
            row=2, col=1
        )
    
    # Complexity over time
    complexity_time = np.cumsum(np.abs(np.diff(signal, axis=0)).mean(axis=1))
    fig.add_trace(
        go.Scatter(
            x=time[1:],
            y=complexity_time,
            mode='lines',
            name='Signal Complexity'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="🌌 Cosmic Whisper: 4D Signal Analysis",
        height=800,
        showlegend=True
    )
    
    return fig

def main():
    """
    Main function to demonstrate the cosmic whisper signal processing pipeline.
    """
    print("🌌 Cosmic Whisper - Quaternion Signal Analysis 🌌")
    print("=" * 50)
    
    # Generate the signal
    print("📡 Generating cosmic signal...")
    signal = get_signal()
    print(f"Signal shape: {signal.shape}")
    
    # Transform to quaternion representation
    print("🔄 Converting to quaternion representation...")
    quaternion_signal = []
    
    # Convert each 3D vector to a quaternion
    for i in range(len(signal)):
        # Normalize the vector to prevent overflow in quaternion conversion
        vector = signal[i]
        norm = np.linalg.norm(vector)
        if norm > 0:
            normalized_vector = vector / norm * min(norm, 1.0)  # Scale down if too large
        else:
            normalized_vector = vector
        
        q = transform_signal(normalized_vector)
        quaternion_signal.append(q)
    
    print(f"Quaternion signal length: {len(quaternion_signal)}")
    
    # Perform Fourier analysis on quaternion data
    print("🔍 Performing quaternion Fourier analysis...")
    fourier_results = quaternion_over_fourier(quaternion_signal)
    
    # Find dominant frequencies
    combined_magnitude = np.sqrt(
        fourier_results['magnitudes']['w']**2 + 
        fourier_results['magnitudes']['x']**2 + 
        fourier_results['magnitudes']['y']**2 + 
        fourier_results['magnitudes']['z']**2
    )
    
    # Get the most significant frequencies (excluding DC component)
    freq_indices = np.argsort(combined_magnitude[1:])[-5:] + 1  # Top 5 frequencies
    dominant_freqs = fourier_results['frequencies'][freq_indices]
    dominant_magnitudes = combined_magnitude[freq_indices]
    
    print("\n🎵 Dominant frequencies detected:")
    for freq, mag in zip(dominant_freqs, dominant_magnitudes):
        print(f"  Frequency: {freq:.4f}, Magnitude: {mag:.2f}")
      # Plot results
    print("\n📊 Generating visualizations...")
    plot_results(signal, quaternion_signal, fourier_results)
    
    print("\n✨ Analysis complete! The cosmic whispers have been decoded. ✨")

    # Advanced signal generation and analysis
    print("\n🔮 Generating advanced signals for comparison...")
    fractal_signal = generate_fractal_signal(len(signal), 3)
    prime_signal = generate_prime_sequence_signal(len(signal), 3)
    chaotic_signal = generate_chaotic_signal(len(signal), 3)
    
    # Convert advanced signals to quaternions
    def safe_transform_signal(sig):
        quaternion_sig = []
        for i in range(len(sig)):
            vector = sig[i]
            norm = np.linalg.norm(vector)
            if norm > 0:
                normalized_vector = vector / norm * min(norm, 1.0)
            else:
                normalized_vector = vector
            q = transform_signal(normalized_vector)
            quaternion_sig.append(q)
        return quaternion_sig
    
    fractal_quaternions = safe_transform_signal(fractal_signal)
    prime_quaternions = safe_transform_signal(prime_signal)
    chaotic_quaternions = safe_transform_signal(chaotic_signal)
    
    # Decode and analyze advanced signals
    print("🤖 Analyzing signals with AI...")
    ai_results_signal = decode_signal_with_ai(signal)
    ai_results_fractal = decode_signal_with_ai(fractal_signal)
    ai_results_prime = decode_signal_with_ai(prime_signal)
    ai_results_chaotic = decode_signal_with_ai(chaotic_signal)
    
    # Complexity analysis
    print("📈 Analyzing signal complexity...")
    complexity_signal = analyze_signal_complexity(signal)
    complexity_fractal = analyze_signal_complexity(fractal_signal)
    complexity_prime = analyze_signal_complexity(prime_signal)
    complexity_chaotic = analyze_signal_complexity(chaotic_signal)
    
    # Print complexity results
    print("\n🧬 Signal Complexity Analysis:")
    print(f"Original Signal - Shannon Entropy: {complexity_signal['dim_0']['shannon_entropy']:.3f}")
    print(f"Fractal Signal - Shannon Entropy: {complexity_fractal['dim_0']['shannon_entropy']:.3f}")
    print(f"Prime Signal - Shannon Entropy: {complexity_prime['dim_0']['shannon_entropy']:.3f}")
    print(f"Chaotic Signal - Shannon Entropy: {complexity_chaotic['dim_0']['shannon_entropy']:.3f}")
    
    # 4D Visualizations
    print("🌌 Creating 4D visualizations...")
    fig_signal = create_4d_visualization(signal, quaternion_signal, ai_results_signal)
    fig_fractal = create_4d_visualization(fractal_signal, fractal_quaternions, ai_results_fractal)
    fig_prime = create_4d_visualization(prime_signal, prime_quaternions, ai_results_prime)
    fig_chaotic = create_4d_visualization(chaotic_signal, chaotic_quaternions, ai_results_chaotic)
      # Show one of the figures as an example
    print("🎭 Displaying original signal 4D visualization...")
    if PLOTLY_AVAILABLE:
        fig_signal.show()
    else:
        print("📊 Matplotlib visualizations displayed above.")
    
    # Signal interpretation
    print("\n🌠 Signal Interpretation Results:")
    signals_data = {
        'Original': {'ai': ai_results_signal, 'complexity': complexity_signal},
        'Fractal': {'ai': ai_results_fractal, 'complexity': complexity_fractal},
        'Prime': {'ai': ai_results_prime, 'complexity': complexity_prime},
        'Chaotic': {'ai': ai_results_chaotic, 'complexity': complexity_chaotic}
    }
    
    for name, data in signals_data.items():
        clusters = len(np.unique(data['ai']['cluster_labels']))
        anomalies = len([x for x in data['ai']['anomaly_labels'] if x == -1])
        reconstruction_loss = data['ai']['reconstruction_loss']
        
        print(f"\n{name} Signal:")
        print(f"  🔍 Clusters detected: {clusters}")
        print(f"  ⚠️  Anomalies found: {anomalies}")
        print(f"  📉 Reconstruction loss: {reconstruction_loss:.4f}")
        print(f"  🌀 Average entropy: {np.mean([d['shannon_entropy'] for d in data['complexity'].values()]):.3f}")
    
    print("\n🚀 All tasks completed. The cosmic whispers and their alien counterparts have been fully analyzed and visualized.")

if __name__ == "__main__":
    main()
