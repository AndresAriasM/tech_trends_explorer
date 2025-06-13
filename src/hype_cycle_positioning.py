# src/hype_cycle_positioning.py
import numpy as np
from typing import List, Dict, Tuple
import math
from decimal import Decimal, InvalidOperation

class HypeCyclePositioner:
    """Maneja el posicionamiento de tecnologías en la curva de Hype Cycle"""
    
    def __init__(self):
        # Definir puntos clave de la curva (basado en Gartner oficial)
        self.phase_positions = {
            "Innovation Trigger": {"x": 8, "y": 25},
            "Peak of Inflated Expectations": {"x": 25, "y": 85},
            "Trough of Disillusionment": {"x": 55, "y": 20},
            "Slope of Enlightenment": {"x": 75, "y": 50},
            "Plateau of Productivity": {"x": 90, "y": 65},
            "Pre-Innovation Trigger": {"x": 3, "y": 10},
            "Unknown": {"x": 50, "y": 50}
        }
        
        # Colores para diferentes tipos de tecnología (basado en tiempo al plateau)
        self.time_colors = {
            "<2 yrs": "#E3F2FD",      # Azul claro
            "2-5 yrs": "#2196F3",     # Azul
            "5-10 yrs": "#1976D2",    # Azul oscuro
            ">10 yrs": "#0D47A1",     # Azul muy oscuro
            "Already there": "#4CAF50" # Verde
        }
    
    def calculate_position(self, phase: str, confidence: float, total_mentions: int = 0) -> Tuple[float, float]:
        """
        Calcula posición exacta basada en fase, confianza y menciones
        
        Args:
            phase: Fase del Hype Cycle
            confidence: Confianza del análisis (0-1)
            total_mentions: Total de menciones encontradas
            
        Returns:
            Tuple con posición (x, y) en la curva
        """
        base_pos = self.phase_positions.get(phase, self.phase_positions["Unknown"])
        
        # NUEVO: Asegurar que los valores son float/int antes de operaciones matemáticas
        try:
            confidence = float(confidence) if confidence is not None else 0.5
            total_mentions = int(total_mentions) if total_mentions is not None else 0
        except (ValueError, TypeError, decimal.InvalidOperation):
            confidence = 0.5
            total_mentions = 0
        
        # Variación basada en confianza (±10% de la posición base)
        confidence_factor = (confidence - 0.5) * 0.2  # -0.1 a +0.1
        
        # Variación basada en menciones (tecnologías con más menciones más al centro)
        mention_factor = min(total_mentions / 1000, 0.1) if total_mentions > 0 else 0
        
        # Añadir algo de aleatoriedad para evitar superposición exacta
        random_x = np.random.uniform(-3, 3)
        random_y = np.random.uniform(-3, 3)
        
        final_x = float(base_pos["x"]) + (confidence_factor * 10) + random_x
        final_y = float(base_pos["y"]) + (confidence_factor * 15) + (mention_factor * 5) + random_y
        
        # Asegurar que esté dentro de los límites
        final_x = max(1, min(98, final_x))
        final_y = max(5, min(95, final_y))
        
        return (float(final_x), float(final_y))
    
    def avoid_overlap(self, technologies: List[Dict]) -> List[Dict]:
        """
        Evita superposición de tecnologías en la gráfica usando algoritmo de separación
        
        Args:
            technologies: Lista de tecnologías con sus posiciones
            
        Returns:
            Lista de tecnologías con posiciones ajustadas
        """
        min_distance = 12  # Distancia mínima entre puntos
        max_iterations = 50  # Evitar bucles infinitos
        
        for iteration in range(max_iterations):
            moved = False
            
            for i, tech1 in enumerate(technologies):
                for j, tech2 in enumerate(technologies[i+1:], i+1):
                    distance = self._calculate_distance(tech1, tech2)
                    
                    if distance < min_distance:
                        # Calcular vector de separación
                        dx = tech2["position_x"] - tech1["position_x"]
                        dy = tech2["position_y"] - tech1["position_y"]
                        
                        # Evitar división por cero
                        if distance == 0:
                            angle = np.random.uniform(0, 2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            distance = 1
                        
                        # Normalizar y escalar
                        separation_factor = (min_distance - distance) / distance / 2
                        move_x = dx * separation_factor
                        move_y = dy * separation_factor
                        
                        # Mover ambas tecnologías en direcciones opuestas
                        tech1["position_x"] -= move_x
                        tech1["position_y"] -= move_y
                        tech2["position_x"] += move_x
                        tech2["position_y"] += move_y
                        
                        # Mantener dentro de límites
                        for tech in [tech1, tech2]:
                            tech["position_x"] = max(1, min(98, tech["position_x"]))
                            tech["position_y"] = max(5, min(95, tech["position_y"]))
                        
                        moved = True
            
            # Si no se movió nada en esta iteración, hemos terminado
            if not moved:
                break
        
        return technologies
    
    def _calculate_distance(self, tech1: Dict, tech2: Dict) -> float:
        """Calcula distancia euclidiana entre dos tecnologías"""
        dx = tech1.get("position_x", 0) - tech2.get("position_x", 0)
        dy = tech1.get("position_y", 0) - tech2.get("position_y", 0)
        return math.sqrt(dx*dx + dy*dy)
    
    def get_color_for_time_to_plateau(self, time_estimate: str) -> str:
        """
        Retorna color basado en tiempo estimado al plateau
        
        Args:
            time_estimate: Estimación de tiempo ("2-5 años", etc.)
            
        Returns:
            Código de color hex
        """
        # Simplificar string de tiempo
        if "ya alcanzado" in time_estimate.lower() or "already" in time_estimate.lower():
            return self.time_colors["Already there"]
        elif any(x in time_estimate.lower() for x in ["<2", "menos de 2", "1-2"]):
            return self.time_colors["<2 yrs"]
        elif any(x in time_estimate.lower() for x in ["2-5", "3-5", "2-4"]):
            return self.time_colors["2-5 yrs"]
        elif any(x in time_estimate.lower() for x in ["5-10", "6-10", "5-8"]):
            return self.time_colors["5-10 yrs"]
        elif any(x in time_estimate.lower() for x in [">10", "más de 10", "10+"]):
            return self.time_colors[">10 yrs"]
        else:
            return self.time_colors["2-5 yrs"]  # Default
    
    def create_hype_cycle_curve(self, x_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera los puntos de la curva característica del Hype Cycle
        
        Args:
            x_points: Número de puntos en la curva
            
        Returns:
            Tuple con arrays de x e y para la curva
        """
        x = np.linspace(0, 100, x_points)
        
        # Curva basada en la forma oficial de Gartner
        y = (
            50 * np.exp(-((x-25)/8)**2) -      # Pico de expectativas (más pronunciado)
            15 * np.exp(-((x-55)/20)**2) +     # Valle de desilusión (más suave)
            30 * np.exp(-((x-90)/12)**2) +     # Plateau de productividad
            10  # Línea base
        )
        
        # Suavizar y ajustar
        y = np.maximum(y, 5)  # Valor mínimo
        y = np.minimum(y, 95)  # Valor máximo
        
        return x, y
    
    def get_phase_label_positions(self) -> List[Dict]:
        """
        Retorna posiciones para las etiquetas de fases en la gráfica
        
        Returns:
            Lista de diccionarios con posiciones y textos de etiquetas
        """
        return [
            {
                "name": "Innovation<br>Trigger", 
                "x": 8, 
                "y": 0,
                "anchor": "top"
            },
            {
                "name": "Peak of Inflated<br>Expectations", 
                "x": 25, 
                "y": 0,
                "anchor": "top"
            },
            {
                "name": "Trough of<br>Disillusionment", 
                "x": 55, 
                "y": 0,
                "anchor": "top"
            },
            {
                "name": "Slope of<br>Enlightenment", 
                "x": 75, 
                "y": 0,
                "anchor": "top"
            },
            {
                "name": "Plateau of<br>Productivity", 
                "x": 90, 
                "y": 0,
                "anchor": "top"
            }
        ]
    
    def estimate_time_to_plateau(self, phase: str, confidence: float = 0.5) -> str:
        """
        Estima tiempo hasta llegar al plateau basado en la fase actual
        
        Args:
            phase: Fase actual del Hype Cycle
            confidence: Confianza del análisis
            
        Returns:
            String con estimación de tiempo
        """
        base_estimates = {
            "Innovation Trigger": "5-10 años",
            "Peak of Inflated Expectations": "2-5 años", 
            "Trough of Disillusionment": "2-5 años",
            "Slope of Enlightenment": "<2 años",
            "Plateau of Productivity": "Ya alcanzado",
            "Pre-Innovation Trigger": ">10 años"
        }
        
        base_time = base_estimates.get(phase, "Incierto")
        
        # Ajustar basado en confianza
        if confidence < 0.3:
            return f"{base_time} (baja confianza)"
        elif confidence > 0.8:
            return base_time
        else:
            return f"{base_time} (estimación)"