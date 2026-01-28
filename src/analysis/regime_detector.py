"""Market Regime Detector using Hidden Markov Model (HMM)
==========================================================

Layer 2 of 6-Layer Architecture

Function: Probabilistically classify market conditions.
This is the PRIMARY FILTER - determines WHEN to trade.

3 States:
- BULLISH: Probability > 60% = Only look for BUY
- BEARISH: Probability > 60% = Only look for SELL
- SIDEWAYS: Probability > 60% = NO TRADE

Input: Cleaned data from Kalman Filter
Output: Regime with probability score

Author: SURIOTA Team
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
from loguru import logger

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed. Install with: pip install hmmlearn")


class MarketRegime(Enum):
    """Market regime types - simplified to 3 states"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeInfo:
    """Regime detection result with probabilities"""
    regime: MarketRegime
    probability: float  # Probability of current regime (0-1)
    probabilities: Dict[str, float]  # All regime probabilities
    bias: str  # 'BUY', 'SELL', or 'NONE'

    @property
    def confidence(self) -> float:
        """Alias for probability (backward compat)"""
        return self.probability

    @property
    def is_tradeable(self) -> bool:
        """Check if regime is suitable for trading"""
        return self.regime != MarketRegime.SIDEWAYS and self.probability >= 0.6

    @property
    def should_buy_only(self) -> bool:
        """Should only look for BUY setups"""
        return self.regime == MarketRegime.BULLISH and self.probability >= 0.6

    @property
    def should_sell_only(self) -> bool:
        """Should only look for SELL setups"""
        return self.regime == MarketRegime.BEARISH and self.probability >= 0.6

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "regime": self.regime.value,
            "probability": self.probability,
            "probabilities": self.probabilities,
            "bias": self.bias,
            "is_tradeable": self.is_tradeable,
        }


class HMMRegimeDetector:
    """Hidden Markov Model based market regime detector

    Uses Gaussian HMM to model market states:
    - State 0: BULLISH (positive returns, moderate volatility)
    - State 1: BEARISH (negative returns, moderate volatility)
    - State 2: SIDEWAYS (near-zero returns, low/high volatility)

    Features:
    - Returns (from Kalman smoothed data)
    - Volatility (rolling std of returns)
    """

    def __init__(
        self,
        n_states: int = 3,
        lookback: int = 50,
        min_samples: int = 100,
        retrain_every: int = 200,
        min_probability: float = 0.6
    ):
        """Initialize HMM regime detector

        Args:
            n_states: Number of hidden states (default 3)
            lookback: Bars to look back for feature calculation
            min_samples: Minimum samples before training
            retrain_every: Retrain model every N updates
            min_probability: Minimum probability to act
        """
        self.n_states = n_states
        self.lookback = lookback
        self.min_samples = min_samples
        self.retrain_every = retrain_every
        self.min_probability = min_probability

        # HMM model
        self.model: Optional[hmm.GaussianHMM] = None
        self._is_trained = False

        # Data storage
        self.price_history: List[float] = []
        self.returns_history: List[float] = []
        self.volatility_history: List[float] = []

        # State mapping (learned from data characteristics)
        self.state_to_regime: Dict[int, MarketRegime] = {}

        # Normalization parameters
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

        self._update_count = 0
        self._last_info: Optional[RegimeInfo] = None

    def _calculate_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate features for HMM

        Returns:
            Tuple of (current_features, historical_features)
        """
        if len(self.price_history) < self.lookback:
            return np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]])

        prices = np.array(self.price_history)

        # Calculate returns
        returns = np.diff(prices) / prices[:-1]

        # Calculate rolling volatility (5-period std)
        volatility = np.zeros(len(returns))
        for i in range(5, len(returns)):
            volatility[i] = np.std(returns[i-5:i])

        # Current features
        current = np.array([[returns[-1], volatility[-1]]])

        # Historical features for training
        historical = np.column_stack([returns[5:], volatility[5:]])

        return current, historical

    def _normalize_features(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize features to zero mean and unit variance

        Args:
            features: Raw features array

        Returns:
            Tuple of (normalized_features, means, stds)
        """
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)

        # Prevent division by zero
        stds = np.where(stds < 1e-8, 1.0, stds)

        normalized = (features - means) / stds
        return normalized, means, stds

    def _train(self):
        """Train HMM model on historical data"""
        if not HMM_AVAILABLE:
            logger.warning("hmmlearn not available, using fallback classifier")
            return

        if len(self.price_history) < self.min_samples:
            return

        _, features = self._calculate_features()

        if len(features) < self.min_samples:
            return

        # Check for valid data variance
        feature_std = np.std(features, axis=0)
        if np.any(feature_std < 1e-8):
            logger.debug("Features have near-zero variance, skipping HMM training")
            return

        # Normalize features for better HMM training
        normalized_features, self._feature_means, self._feature_stds = self._normalize_features(features)

        # Suppress convergence warnings during training
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                # Initialize HMM with diagonal covariance (more stable)
                self.model = hmm.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="diag",  # Use diagonal for stability
                    n_iter=100,
                    random_state=42,
                    init_params="stmc",  # Initialize all params
                    params="stmc"
                )

                # Add small regularization to prevent singular covariance
                self.model.min_covar = 1e-3

                # Fit model on normalized features
                self.model.fit(normalized_features)

                # Map states to regimes based on mean returns
                self._map_states_to_regimes(normalized_features)

                self._is_trained = True
                logger.debug(f"HMM trained on {len(features)} samples")

            except Exception as e:
                logger.warning(f"HMM training failed: {e}")
                self._is_trained = False

    def _map_states_to_regimes(self, features: np.ndarray):
        """Map HMM states to market regimes based on feature characteristics"""
        if self.model is None:
            return

        # Get state sequence
        states = self.model.predict(features)

        # Calculate mean returns for each state
        state_returns = {}
        for state in range(self.n_states):
            mask = states == state
            if np.sum(mask) > 0:
                state_returns[state] = np.mean(features[mask, 0])
            else:
                state_returns[state] = 0.0

        # Sort states by return
        sorted_states = sorted(state_returns.items(), key=lambda x: x[1])

        # Map: lowest return = BEARISH, highest = BULLISH, middle = SIDEWAYS
        if len(sorted_states) >= 3:
            self.state_to_regime[sorted_states[0][0]] = MarketRegime.BEARISH
            self.state_to_regime[sorted_states[1][0]] = MarketRegime.SIDEWAYS
            self.state_to_regime[sorted_states[2][0]] = MarketRegime.BULLISH
        elif len(sorted_states) == 2:
            self.state_to_regime[sorted_states[0][0]] = MarketRegime.BEARISH
            self.state_to_regime[sorted_states[1][0]] = MarketRegime.BULLISH
        else:
            self.state_to_regime[0] = MarketRegime.UNKNOWN

    def update(self, price: float) -> RegimeInfo:
        """Update detector with new price

        Args:
            price: Current close price (preferably Kalman smoothed)

        Returns:
            RegimeInfo with current regime and probabilities
        """
        self.price_history.append(price)

        # Keep limited history
        if len(self.price_history) > 1000:
            self.price_history = self.price_history[-1000:]

        self._update_count += 1

        # Retrain periodically
        if self._update_count % self.retrain_every == 0:
            self._train()

        # If not trained, use fallback
        if not self._is_trained or self.model is None:
            return self._fallback_detection()

        # Get current features
        current_features, _ = self._calculate_features()

        # Normalize current features using training params
        if self._feature_means is not None and self._feature_stds is not None:
            current_features = (current_features - self._feature_means) / self._feature_stds

        try:
            # Predict state probabilities
            state_probs = self.model.predict_proba(current_features)[0]

            # Find most likely state
            current_state = np.argmax(state_probs)
            regime = self.state_to_regime.get(current_state, MarketRegime.UNKNOWN)
            probability = state_probs[current_state]

            # Build probabilities dict
            probs_dict = {}
            for state, reg in self.state_to_regime.items():
                if state < len(state_probs):
                    probs_dict[reg.value] = float(state_probs[state])

            # Determine bias
            if regime == MarketRegime.BULLISH and probability >= self.min_probability:
                bias = 'BUY'
            elif regime == MarketRegime.BEARISH and probability >= self.min_probability:
                bias = 'SELL'
            else:
                bias = 'NONE'

            self._last_info = RegimeInfo(
                regime=regime,
                probability=probability,
                probabilities=probs_dict,
                bias=bias
            )

        except Exception as e:
            logger.warning(f"HMM prediction failed: {e}")
            return self._fallback_detection()

        return self._last_info

    def _fallback_detection(self) -> RegimeInfo:
        """Simple fallback when HMM not available/trained"""
        if len(self.price_history) < 20:
            return RegimeInfo(
                regime=MarketRegime.UNKNOWN,
                probability=0.0,
                probabilities={'UNKNOWN': 1.0},
                bias='NONE'
            )

        # Simple trend detection using linear regression
        recent = np.array(self.price_history[-20:])
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]

        # Normalize slope
        price_range = max(recent) - min(recent)
        if price_range < 0.0001:
            regime = MarketRegime.SIDEWAYS
            prob = 0.8
        else:
            normalized_slope = slope / price_range * 100

            if normalized_slope > 0.5:
                regime = MarketRegime.BULLISH
                prob = min(0.9, 0.5 + abs(normalized_slope) * 0.1)
            elif normalized_slope < -0.5:
                regime = MarketRegime.BEARISH
                prob = min(0.9, 0.5 + abs(normalized_slope) * 0.1)
            else:
                regime = MarketRegime.SIDEWAYS
                prob = 0.7

        # Determine bias
        if regime == MarketRegime.BULLISH and prob >= self.min_probability:
            bias = 'BUY'
        elif regime == MarketRegime.BEARISH and prob >= self.min_probability:
            bias = 'SELL'
        else:
            bias = 'NONE'

        self._last_info = RegimeInfo(
            regime=regime,
            probability=prob,
            probabilities={regime.value: prob},
            bias=bias
        )

        return self._last_info

    def get_trading_bias(self) -> Tuple[str, float]:
        """Get current trading bias

        Returns:
            Tuple of (bias: 'BUY'/'SELL'/'NONE', confidence)
        """
        if self._last_info is None:
            return 'NONE', 0.0

        return self._last_info.bias, self._last_info.probability

    def should_trade(self) -> bool:
        """Check if current regime allows trading"""
        if self._last_info is None:
            return False
        return self._last_info.is_tradeable

    @property
    def last_info(self) -> Optional[RegimeInfo]:
        """Get last regime info"""
        return self._last_info

    def reset(self):
        """Reset detector state"""
        self.price_history.clear()
        self.returns_history.clear()
        self.volatility_history.clear()
        self._update_count = 0
        self._last_info = None
        self._is_trained = False
        self.model = None
        self.state_to_regime.clear()
        self._feature_means = None
        self._feature_stds = None
