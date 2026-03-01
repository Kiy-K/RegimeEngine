import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from gravitas_engine.agents.rl_env import RegimeEnv
from gravitas_engine.agents.survival_env import SurvivalRegimeEnv
from gravitas_engine.agents.gravitas_env import GravitasEnv

def test_single_env(env_class, name):
    print(f"Checking {name}...")
    try:
        env = env_class()
        check_env(env)
        print(f"  {name} check passed!")
    except Exception as e:
        print(f"  {name} check FAILED: {e}")
        # traceback 
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_single_env(RegimeEnv, "RegimeEnv")
    test_single_env(SurvivalRegimeEnv, "SurvivalRegimeEnv")
    test_single_env(GravitasEnv, "GravitasEnv")
    print("\nAll environment checks passed!")
