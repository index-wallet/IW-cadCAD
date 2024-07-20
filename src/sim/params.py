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
sim_timesteps = 30

# Reward size for 1 unit of donation utility, before any donations have been made
initial_donation_reward_amount: float = 2
# Fraction of a currency reward that is returned in the original wallet
# This is equivalent to preventing agents from donating any more than 1-donation_reward_mix
# of their wallet to any individual good, which is how I implement this
donation_reward_mix: float = 0.9
