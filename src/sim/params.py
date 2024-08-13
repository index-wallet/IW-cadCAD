from typing import List

grid_size: int = 10
edge_prob: float = 0.8
wraparound: bool = True

num_currencies: int = 3

# Each of these ranges is [min, width] for a uniform sample
demand_range: List[float] = [0.5, 1]
valuation_range: List[float] = [0.5, 1]
price_range: List[float] = [0.1, 0.9]

eps: float = 10**-6
sim_timesteps = 30

## Debugging flag; "try's to" log additional information and also determines the execution context for cadCAD 
is_debug: bool = True

# Reward size for 1 unit of donation utility, before any donations have been made
initial_donation_reward_amount: float = 2
# Fraction of a currency reward that is returned in the original wallet
# This is equivalent to preventing agents from donating any more than 1-donation_reward_mix
# of their wallet to any individual good, which is how I implement this
donation_reward_mix: float = 0.9

## Type of topology to use for the simulation
## grid: simple grid topology; all nodes have 4 or less neighbors (at least 1) and are both buyers and sellers
## vendor_customer: vendors are sellers and customers are buyers; All venders connect to all customers and vice versa; no self-interactions; hardcoded 1:5 ratio
## TODO: Add customizable ratio
topology_type: str = "grid"

## If `topology_type` is `vendor_customer`, this is the ratio of customers to vendors. Must be at least `1`. If for example `5`, there are 5 customers for every vendor.
vendor_customer_ratio: int = 5