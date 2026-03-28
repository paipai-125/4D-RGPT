import torch
from typing import Dict, Any

class DependencyAwareWeighter:
    def __init__(self, momentum: float = 0.99, min_weight: float = 0.1):
        self.momentum = momentum
        self.min_weight = min_weight
        # Global states lambda for quality normalization
        self.lambda_map = {}
        # Stores running ema of qualities
        self.quality_ema = {
            'seg': 0.1,        # initialize low
            'obj_depth': 0.1,
            'obj_flow': 0.1,
            'obj_mo': 0.1,
            'orient': 0.1
        }
        
    def _compute_quality(self, task: str, value: float) -> float:
        if task == 'seg':
            # value is average dice directly
            return min(max(value, 0.0), 1.0)
        else:
            # tasks are obj_depth, obj_flow, obj_mo, etc.
            # Use λ / (loss + λ) mapping 
            # where loss is the `value` and lambda is median dynamically initialized
            
            if task not in self.lambda_map:
                # Store the very first loss as an approximation for median lambda
                self.lambda_map[task] = value + 1e-6
            
            lam = self.lambda_map[task]
            q = lam / (value + lam)
            return min(max(q, 0.0), 1.0)
            
    def compute_weight(self, task: str, stat_value: float) -> float:
        current_quality = self._compute_quality(task, stat_value)
        
        # EMA
        if task not in self.quality_ema:
            self.quality_ema[task] = current_quality
        else:
            self.quality_ema[task] = self.momentum * self.quality_ema[task] + (1 - self.momentum) * current_quality
        
        # Map to weight [min_weight, 1.0]
        weight = self.min_weight + (1.0 - self.min_weight) * self.quality_ema[task]
        return weight
        
    def get_quality(self, task: str) -> float:
        return self.quality_ema.get(task, self.min_weight)
        
    def state_dict(self):
        return {
            'quality_ema': self.quality_ema,
            'lambda_map': self.lambda_map
        }
        
    def load_state_dict(self, state):
        self.quality_ema = state.get('quality_ema', self.quality_ema)
        self.lambda_map = state.get('lambda_map', self.lambda_map)

class DependencyWeightManager:
    def __init__(self, weighter: DependencyAwareWeighter):
        self.weighter = weighter
        # Define dependencies. 
        # Since seg doesn't strictly have a dependency that blocks it entirely right now without explicit global output, we can omit upstream or just let it spin independently
        self.dependency_graph = {
            'seg': {
                'upstream': [], # 'obj_depth', 'obj_flow', 'obj_mo' combined might serve but since you said seg loss doesn't strictly depend on L4P, we'll keep it as base quality multiplier for downstream or just follow instruction 3
            },
            'orient': {
                'upstream': ['seg'],
                'weight_rule': 'direct'
            },
            'obj_depth': {
                'upstream': ['seg'],
                'weight_rule': 'direct'
            },
            'obj_flow': {
                'upstream': ['seg'],
                'weight_rule': 'direct'
            },
            'obj_mo': {
                'upstream': ['seg'],
                'weight_rule': 'direct'
            }
        }
        
    def compute_task_weight(self, task_name: str) -> float:
        if task_name not in self.dependency_graph:
            return 1.0
            
        config = self.dependency_graph[task_name]
        upstreams = config.get('upstream', [])
        if not upstreams:
            return 1.0
            
        rule = config.get('weight_rule', 'direct')
        
        qualities = []
        for upstream in upstreams:
            # We map quality [0, 1] to weight scale [min_weight, 1.0]
            # Since seg provides the quality, we map the seg's current EMA quality to an actual weight coefficient
            q = self.weighter.get_quality(upstream)
            w = self.weighter.min_weight + (1.0 - self.weighter.min_weight) * q
            qualities.append(w)
            
        if rule == 'direct':
            return qualities[0]
        elif rule == 'product':
            # For multiple upstreams 
            prod = 1.0
            for w in qualities:
                prod *= w
            return prod
            
        return 1.0
        
    def step(self, task_stats: Dict[str, float]) -> Dict[str, float]:
        """
        task_stats: e.g., {'seg': dice_score, 'obj_depth': loss_depth_val, ...}
        """
        # First update all EMAs
        for task, value in task_stats.items():
            self.weighter.compute_weight(task, value)
            
        # Then compute current step weights for each downstream task
        weights = {}
        for task in self.dependency_graph.keys():
            weights[task] = self.compute_task_weight(task)
            
        # Add weights for implicit variables if asked
        return weights

