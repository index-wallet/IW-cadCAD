from typing import List

grid_size: int = 10
edge_prob: float = 0.8
wraparound: bool = True

num_currencies: int = 2

# Each of these ranges is [min, width] for a uniform sample
demand_range: List[float] = [0.5, 1]
valuation_range: List[float] = [0.5, 1]
price_range: List[float] = [0.1, 0.9]

eps: float = 10**-6

initial_donation_reward_amount = 2
