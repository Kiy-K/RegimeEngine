from gymnasium.utils.env_checker import check_env
from regime_engine.agents.rl_env import RegimeEnv

env = RegimeEnv()
check_env(env)
print("Environment check passed!")
