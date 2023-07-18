from pandia.agent.env import Action, WebRTCEnv, Observation
from pandia.log_analyzer import CODEC_NAMES


def main():
    env = WebRTCEnv(config={
        'sender_log': '/tmp/pandia-sender.log',
    })
    obs, info = env.reset()
    action = Action()
    action.fps[0] = 10
    action.resolution[0] = 720
    action.bitrate[0] = 1024
    action.pacing_rate[0] = 500 * 1024
    for i in range(30):
        if i == 10:
            action.resolution[0] = 2160
        elif i == 20:
            action.resolution[0] = 240
        obs, reward, done, truncated, info = env.step(action.array())
        observation = Observation.from_array(obs)
        if done or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()
