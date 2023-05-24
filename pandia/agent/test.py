from pandia.agent.env import Action, WebRTCEnv, Observation


def main():
    env = WebRTCEnv({})
    observation, info = env.reset(seed=42)
    action = Action()
    action.fps = 10
    action.resolution = 720
    action.bitrate = 1024
    action.pacing_rate = 500 * 1024
    obs, info = env.reset()
    for _ in range(30):
        obs, reward, done, truncated, info = env.step(action.array())
        observation = Observation.from_array(obs)
        print(f'Step: {env.step_count}, Reward: {reward}, '
              f'G2G delay: {observation.frame_g2g_delay}')
        if done or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()
