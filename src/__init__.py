"""SURGE-WSI - Quantitative Trading System with Smart Money Concepts

6-Layer Architecture:
1. Kalman Filter - Noise reduction
2. HMM Regime Detection - Market state classification
3. Kill Zone - Time filter (ICT sessions)
4. POI Detection - Order Blocks + FVG
5. Entry Trigger - LTF confirmation
6. Exit Management - Partial TP + trailing
"""

__version__ = "1.0.0"
__author__ = "SURIOTA Team"
