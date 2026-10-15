"""
backend/core/positioning.py
Ported from src/hype_cycle_positioning.py — pure numpy, no Streamlit/Plotly.
"""

import math
from decimal import InvalidOperation
from typing import Dict, List, Tuple

import numpy as np


class HypeCyclePositioner:
    """Positions a technology on the Gartner Hype Cycle curve."""

    phase_positions: Dict[str, Dict[str, int]] = {
        "Innovation Trigger":             {"x": 8,  "y": 25},
        "Peak of Inflated Expectations":  {"x": 25, "y": 85},
        "Trough of Disillusionment":      {"x": 55, "y": 20},
        "Slope of Enlightenment":         {"x": 75, "y": 50},
        "Plateau of Productivity":        {"x": 90, "y": 65},
        "Pre-Innovation Trigger":         {"x": 3,  "y": 10},
        "Unknown":                        {"x": 50, "y": 50},
    }

    time_colors: Dict[str, str] = {
        "<2 yrs":       "#E3F2FD",
        "2-5 yrs":      "#2196F3",
        "5-10 yrs":     "#1976D2",
        ">10 yrs":      "#0D47A1",
        "Already there":"#4CAF50",
    }

    def calculate_position(
        self, phase: str, confidence: float, total_mentions: int = 0
    ) -> Tuple[float, float]:
        base_pos = self.phase_positions.get(phase, self.phase_positions["Unknown"])
        try:
            confidence = float(confidence) if confidence is not None else 0.5
            total_mentions = int(total_mentions) if total_mentions is not None else 0
        except (ValueError, TypeError, InvalidOperation):
            confidence = 0.5
            total_mentions = 0

        confidence_factor = (confidence - 0.5) * 0.2
        mention_factor = min(total_mentions / 1000, 0.1) if total_mentions > 0 else 0
        random_x = np.random.uniform(-3, 3)
        random_y = np.random.uniform(-3, 3)

        final_x = float(base_pos["x"]) + (confidence_factor * 10) + random_x
        final_y = float(base_pos["y"]) + (confidence_factor * 15) + (mention_factor * 5) + random_y
        final_x = max(1.0, min(98.0, final_x))
        final_y = max(5.0, min(95.0, final_y))
        return final_x, final_y

    def estimate_time_to_plateau(self, phase: str, confidence: float = 0.5) -> str:
        base_estimates = {
            "Innovation Trigger":            "5-10 años",
            "Peak of Inflated Expectations": "2-5 años",
            "Trough of Disillusionment":     "2-5 años",
            "Slope of Enlightenment":        "<2 años",
            "Plateau of Productivity":       "Ya alcanzado",
            "Pre-Innovation Trigger":        ">10 años",
        }
        base_time = base_estimates.get(phase, "Incierto")
        if confidence < 0.3:
            return f"{base_time} (baja confianza)"
        elif confidence > 0.8:
            return base_time
        else:
            return f"{base_time} (estimación)"

    def get_color_for_time_to_plateau(self, time_estimate: str) -> str:
        t = time_estimate.lower()
        if "ya alcanzado" in t or "already" in t:
            return self.time_colors["Already there"]
        elif any(x in t for x in ["<2", "menos de 2", "1-2"]):
            return self.time_colors["<2 yrs"]
        elif any(x in t for x in ["2-5", "3-5", "2-4"]):
            return self.time_colors["2-5 yrs"]
        elif any(x in t for x in ["5-10", "6-10", "5-8"]):
            return self.time_colors["5-10 yrs"]
        elif any(x in t for x in [">10", "más de 10", "10+"]):
            return self.time_colors[">10 yrs"]
        return self.time_colors["2-5 yrs"]
