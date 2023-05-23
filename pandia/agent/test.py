from pandia.agent.env import WebRTCEnv


def main():
    env = WebRTCEnv()
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(
            f'Step: {env.step_count}, Reward: {reward}, Observation: {observation["frame_g2g_delay"]}')
        if done:
            break


if __name__ == "__main__":
    main()
