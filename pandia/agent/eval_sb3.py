from pandia.agent.env import WebRTCEnv
from stable_baselines3 import SAC


def main():
    env = WebRTCEnv(config={'legacy_api': True})
    model = SAC.load("sac_pandia", env)
    


if __name__ == "__main__":
    main()
