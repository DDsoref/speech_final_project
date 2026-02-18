"""
Noise Selectors for Domain Classification

Implements unsupervised clustering-based selectors for determining
which domain (noise type) a test sample belongs to.

Paper Reference: Section III.D - Unsupervised Noise Selector

From paper:
"We devise an unsupervised noise selector capable of accurately 
determining the domain to which an input noisy speech belongs."

The selector uses features from the frozen encoder and clusters them
to identify domains without supervision.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
import pickle
from pathlib import Path


class BaseNoiseSelector(ABC):
    """
    Abstract base class for noise selectors
    
    Paper: Section III.D
    
    The selector operates in two phases:
    1. Training: Cluster features from training data to find domain centers
    2. Inference: Classify test samples by finding nearest cluster
    
    Key innovation: Training-free (unsupervised) - no labels needed!
    """
    
    def __init__(self, feature_dim: int = 256):
        """
        Args:
            feature_dim: Dimension of encoder features
        """
        self.feature_dim = feature_dim
        self.is_fitted = False
        self.cluster_centers = {}  # {session_id: centers}
        self.num_sessions = 0
    
    @abstractmethod
    def fit_session(
        self,
        features: np.ndarray,
        session_id: int
    ) -> np.ndarray:
        """
        Fit selector for a new session's domain
        
        Paper Equation (4):
        K^t = Kmeans(MeanP(E(X^t; φ_E^0)))
        
        Args:
            features: Features from encoder [N, D]
            session_id: Session ID (1, 2, 3, ...)
        
        Returns:
            Cluster centers for this session
        """
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> int:
        """
        Predict which session's domain a sample belongs to
        
        Paper Equation (5):
        j = argmin_ℓ L2(K^ℓ, MeanP[E(z^t_i; φ_E^0)])
        
        Args:
            features: Features from encoder [D] or [N, D]
        
        Returns:
            Predicted session ID
        """
        pass
    
    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Predict session for a batch of samples
        
        Args:
            features: Features [B, D]
        
        Returns:
            Predicted session IDs [B]
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        predictions = np.array([self.predict(f) for f in features])
        return predictions
    
    def save(self, path: str):
        """Save selector state"""
        state = {
            'feature_dim': self.feature_dim,
            'is_fitted': self.is_fitted,
            'cluster_centers': self.cluster_centers,
            'num_sessions': self.num_sessions,
            'selector_type': self.__class__.__name__
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Selector saved: {path}")
    
    def load(self, path: str):
        """Load selector state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.feature_dim = state['feature_dim']
        self.is_fitted = state['is_fitted']
        self.cluster_centers = state['cluster_centers']
        self.num_sessions = state['num_sessions']
        
        print(f"Selector loaded: {path}")


class KMeansSelector(BaseNoiseSelector):
    """
    K-Means based noise selector
    
    Paper Reference: Section III.D, Equation (4)
    
    This is the method used in the paper. It uses K-Means clustering
    to find representative centers for each domain.
    
    From paper:
    "In the training stage of session t, we use the feature extractor 
    E(·; φ_E^0) of the pre-trained model to initialize the domain selector:
    K^t = Kmeans(MeanP(E(X^t; φ_E^0)))"
    
    Paper tests k ∈ {10, 20, 50}, finding k=20 works best (Table III)
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        n_clusters: int = 20,
        random_state: int = 42
    ):
        """
        Args:
            feature_dim: Dimension of encoder features
            n_clusters: Number of clusters (k in paper, default=20)
            random_state: Random seed for reproducibility
        """
        super().__init__(feature_dim)
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_models = {}  # {session_id: KMeans model}
    
    def fit_session(
        self,
        features: np.ndarray,
        session_id: int
    ) -> np.ndarray:
        """
        Fit K-Means for a new session
        
        Paper Equation (4):
        K^t = Kmeans(MeanP(E(X^t; φ_E^0)))
        
        Args:
            features: Features from training data [N, D]
            session_id: Session ID
        
        Returns:
            Cluster centers [k, D]
        """
        print(f"Fitting K-Means for session {session_id}:")
        print(f"  Features shape: {features.shape}")
        print(f"  Number of clusters: {self.n_clusters}")
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(-1, self.feature_dim)
        
        # Fit K-Means
        kmeans = KMeans(
            n_clusters=min(self.n_clusters, len(features)),  # Handle small datasets
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        kmeans.fit(features)
        
        # Store cluster centers
        self.cluster_centers[session_id] = kmeans.cluster_centers_
        self.kmeans_models[session_id] = kmeans
        self.num_sessions = max(self.num_sessions, session_id + 1)
        self.is_fitted = True
        
        print(f"  Fitted {len(kmeans.cluster_centers_)} clusters")
        print(f"  Inertia: {kmeans.inertia_:.2f}")
        
        return kmeans.cluster_centers_
    
    def predict(self, features: np.ndarray) -> int:
        """
        Predict session by finding closest cluster centers
        
        Paper Equation (5):
        j = argmin_ℓ L2(K^ℓ, MeanP[E(z^t_i; φ_E^0)])
        
        Args:
            features: Features from test sample [D]
        
        Returns:
            Predicted session ID
        """
        if not self.is_fitted:
            raise ValueError("Selector not fitted. Call fit_session first.")
        
        # Ensure features are 1D
        if features.ndim > 1:
            features = features.flatten()
        
        # Calculate distance to each session's cluster centers
        min_distance = float('inf')
        best_session = 0
        
        for session_id, centers in self.cluster_centers.items():
            # Calculate L2 distance to all centers, take minimum
            # Paper: "identify the closest cluster center"
            distances = np.linalg.norm(centers - features, axis=1)
            min_dist_to_session = np.min(distances)
            
            if min_dist_to_session < min_distance:
                min_distance = min_dist_to_session
                best_session = session_id
        
        return best_session
    
    def get_selection_probabilities(self, features: np.ndarray) -> Dict[int, float]:
        """
        Get probabilities for each session (for analysis)
        
        Args:
            features: Features from test sample [D]
        
        Returns:
            Dictionary {session_id: probability}
        """
        if not self.is_fitted:
            raise ValueError("Selector not fitted")
        
        if features.ndim > 1:
            features = features.flatten()
        
        # Calculate distances to all sessions
        distances = {}
        for session_id, centers in self.cluster_centers.items():
            dists = np.linalg.norm(centers - features, axis=1)
            distances[session_id] = np.min(dists)
        
        # Convert to probabilities (inverse distance, normalized)
        inv_distances = {s: 1.0 / (d + 1e-8) for s, d in distances.items()}
        total = sum(inv_distances.values())
        probabilities = {s: v / total for s, v in inv_distances.items()}
        
        return probabilities


class MeanShiftSelector(BaseNoiseSelector):
    """
    Mean-Shift based noise selector
    
    YOUR POTENTIAL IMPROVEMENT!
    
    Advantages over K-Means:
    1. No need to specify number of clusters (automatic)
    2. Can find arbitrary-shaped clusters
    3. Robust to outliers
    
    Mean-Shift finds modes in the feature distribution, making it
    potentially better at discovering natural domain boundaries.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        bandwidth: Optional[float] = None,
        quantile: float = 0.3,
        n_samples: int = 500
    ):
        """
        Args:
            feature_dim: Dimension of encoder features
            bandwidth: Kernel bandwidth (None = auto-estimate)
            quantile: Quantile for bandwidth estimation
            n_samples: Number of samples for bandwidth estimation
        """
        super().__init__(feature_dim)
        self.bandwidth = bandwidth
        self.quantile = quantile
        self.n_samples = n_samples
        self.meanshift_models = {}
    
    def fit_session(
        self,
        features: np.ndarray,
        session_id: int
    ) -> np.ndarray:
        """
        Fit Mean-Shift for a new session
        
        Args:
            features: Features from training data [N, D]
            session_id: Session ID
        
        Returns:
            Cluster centers [k, D] (k is determined automatically)
        """
        print(f"Fitting Mean-Shift for session {session_id}:")
        print(f"  Features shape: {features.shape}")
        
        if features.ndim == 1:
            features = features.reshape(-1, self.feature_dim)
        
        # Estimate bandwidth if not provided
        bandwidth = self.bandwidth
        if bandwidth is None:
            # Subsample for efficiency
            n_samples = min(self.n_samples, len(features))
            sample_idx = np.random.choice(len(features), n_samples, replace=False)
            bandwidth = estimate_bandwidth(
                features[sample_idx],
                quantile=self.quantile
            )
            print(f"  Estimated bandwidth: {bandwidth:.4f}")
        
        # Fit Mean-Shift
        meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        meanshift.fit(features)
        
        # Store cluster centers
        self.cluster_centers[session_id] = meanshift.cluster_centers_
        self.meanshift_models[session_id] = meanshift
        self.num_sessions = max(self.num_sessions, session_id + 1)
        self.is_fitted = True
        
        print(f"  Found {len(meanshift.cluster_centers_)} clusters (automatic)")
        
        return meanshift.cluster_centers_
    
    def predict(self, features: np.ndarray) -> int:
        """
        Predict session by finding closest cluster centers
        
        Same logic as K-Means but with different cluster centers
        
        Args:
            features: Features from test sample [D]
        
        Returns:
            Predicted session ID
        """
        if not self.is_fitted:
            raise ValueError("Selector not fitted")
        
        if features.ndim > 1:
            features = features.flatten()
        
        min_distance = float('inf')
        best_session = 0
        
        for session_id, centers in self.cluster_centers.items():
            distances = np.linalg.norm(centers - features, axis=1)
            min_dist_to_session = np.min(distances)
            
            if min_dist_to_session < min_distance:
                min_distance = min_dist_to_session
                best_session = session_id
        
        return best_session


class GMMSelector(BaseNoiseSelector):
    """
    Gaussian Mixture Model based noise selector
    
    ANOTHER POTENTIAL IMPROVEMENT!
    
    Advantages:
    1. Probabilistic model (gives uncertainty estimates)
    2. Soft clustering (samples can belong to multiple clusters)
    3. Models covariance structure
    
    GMM can provide confidence scores for predictions,
    useful for identifying ambiguous samples.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        n_components: int = 20,
        covariance_type: str = 'full',
        random_state: int = 42
    ):
        """
        Args:
            feature_dim: Dimension of encoder features
            n_components: Number of Gaussian components
            covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            random_state: Random seed
        """
        super().__init__(feature_dim)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm_models = {}
    
    def fit_session(
        self,
        features: np.ndarray,
        session_id: int
    ) -> np.ndarray:
        """
        Fit GMM for a new session
        
        Args:
            features: Features from training data [N, D]
            session_id: Session ID
        
        Returns:
            Gaussian means [k, D]
        """
        print(f"Fitting GMM for session {session_id}:")
        print(f"  Features shape: {features.shape}")
        print(f"  Number of components: {self.n_components}")
        
        if features.ndim == 1:
            features = features.reshape(-1, self.feature_dim)
        
        # Fit GMM
        gmm = GaussianMixture(
            n_components=min(self.n_components, len(features)),
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=100
        )
        gmm.fit(features)
        
        # Store means as cluster centers
        self.cluster_centers[session_id] = gmm.means_
        self.gmm_models[session_id] = gmm
        self.num_sessions = max(self.num_sessions, session_id + 1)
        self.is_fitted = True
        
        print(f"  Fitted {len(gmm.means_)} components")
        print(f"  Converged: {gmm.converged_}")
        
        return gmm.means_
    
    def predict(self, features: np.ndarray) -> int:
        """
        Predict session using GMM log-likelihood
        
        For GMM, we use log-likelihood instead of distance
        Higher likelihood = better match to that session's distribution
        
        Args:
            features: Features from test sample [D]
        
        Returns:
            Predicted session ID
        """
        if not self.is_fitted:
            raise ValueError("Selector not fitted")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Calculate log-likelihood for each session's GMM
        max_likelihood = float('-inf')
        best_session = 0
        
        for session_id, gmm in self.gmm_models.items():
            log_likelihood = gmm.score(features)
            
            if log_likelihood > max_likelihood:
                max_likelihood = log_likelihood
                best_session = session_id
        
        return best_session
    
    def get_selection_confidence(self, features: np.ndarray) -> Dict[int, float]:
        """
        Get confidence scores (log-likelihood) for each session
        
        Args:
            features: Features from test sample [D]
        
        Returns:
            Dictionary {session_id: log_likelihood}
        """
        if not self.is_fitted:
            raise ValueError("Selector not fitted")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        confidences = {}
        for session_id, gmm in self.gmm_models.items():
            confidences[session_id] = gmm.score(features)
        
        return confidences


# ============================================================================
# Factory Function
# ============================================================================

def create_selector(
    selector_type: str = "kmeans",
    feature_dim: int = 256,
    **kwargs
) -> BaseNoiseSelector:
    """
    Factory function to create noise selector
    
    Args:
        selector_type: One of ["kmeans", "meanshift", "gmm"]
        feature_dim: Dimension of encoder features
        **kwargs: Selector-specific arguments
    
    Returns:
        Noise selector instance
    """
    if selector_type == "kmeans":
        return KMeansSelector(feature_dim=feature_dim, **kwargs)
    elif selector_type == "meanshift":
        return MeanShiftSelector(feature_dim=feature_dim, **kwargs)
    elif selector_type == "gmm":
        return GMMSelector(feature_dim=feature_dim, **kwargs)
    else:
        raise ValueError(f"Unknown selector type: {selector_type}")


# ============================================================================
# Demo and Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Noise Selectors...")
    
    # Generate dummy features for testing
    np.random.seed(42)
    feature_dim = 256
    
    # Session 0: Cluster around [0, 0, ...]
    features_s0 = np.random.randn(500, feature_dim) * 0.5
    
    # Session 1: Cluster around [2, 2, ...]
    features_s1 = np.random.randn(300, feature_dim) * 0.5 + 2
    
    # Session 2: Cluster around [-2, -2, ...]
    features_s2 = np.random.randn(300, feature_dim) * 0.5 - 2
    
    # Test K-Means Selector
    print("\n1. Testing K-Means Selector:")
    kmeans_selector = KMeansSelector(feature_dim=feature_dim, n_clusters=20)
    
    kmeans_selector.fit_session(features_s0, session_id=0)
    kmeans_selector.fit_session(features_s1, session_id=1)
    kmeans_selector.fit_session(features_s2, session_id=2)
    
    # Test prediction
    test_feature = np.random.randn(feature_dim) * 0.5 + 2  # Should be session 1
    pred = kmeans_selector.predict(test_feature)
    print(f"  Test feature (near session 1) predicted as: session {pred}")
    
    probs = kmeans_selector.get_selection_probabilities(test_feature)
    print(f"  Selection probabilities: {probs}")
    
    # Test Mean-Shift Selector
    print("\n2. Testing Mean-Shift Selector:")
    meanshift_selector = MeanShiftSelector(feature_dim=feature_dim)
    
    meanshift_selector.fit_session(features_s0, session_id=0)
    meanshift_selector.fit_session(features_s1, session_id=1)
    
    pred = meanshift_selector.predict(test_feature)
    print(f"  Test feature predicted as: session {pred}")
    
    # Test GMM Selector
    print("\n3. Testing GMM Selector:")
    gmm_selector = GMMSelector(feature_dim=feature_dim, n_components=10)
    
    gmm_selector.fit_session(features_s0, session_id=0)
    gmm_selector.fit_session(features_s1, session_id=1)
    
    pred = gmm_selector.predict(test_feature)
    print(f"  Test feature predicted as: session {pred}")
    
    confidences = gmm_selector.get_selection_confidence(test_feature)
    print(f"  Confidence scores: {confidences}")
    
    # Test save/load
    print("\n4. Testing save/load:")
    kmeans_selector.save("checkpoints/test_selector.pkl")
    
    new_selector = KMeansSelector(feature_dim=feature_dim)
    new_selector.load("checkpoints/test_selector.pkl")
    print(f"  Loaded selector has {new_selector.num_sessions} sessions")
    
    print("\n✓ All selectors working!")