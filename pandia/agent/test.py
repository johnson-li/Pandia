from pandia.agent.env import Action, WebRTCEnv, Observation
from pandia.log_analyzer import CODEC_NAMES


def main():
    env = WebRTCEnv(config={'sender_log': '/tmp/pandia-sender.log',})
    obs, info = env.reset()
    action = Action()
    action.fps = 10
    action.resolution = 720
    action.bitrate = 800
    action.pacing_rate = 500 * 1024
    for i in range(30):
        if i == 15:
            action.resolution = 400 
        obs, reward, done, truncated, info = env.step(action.array())
        observation = Observation.from_array(obs)
        delays = (observation.frame_encoding_delay[0], 
                  observation.frame_pacing_delay[0], 
                  observation.frame_assemble_delay[0], 
                  observation.frame_g2g_delay[0])
        print(f'Step: {env.step_count}, Reward: {reward:.02f}, '
              f'Delays: {delays}, '
              f'Width: {observation.frame_width[0]}/{observation.frame_encoded_width[0]}, '
              f'FPS: {observation.fps[0]}, '
              f'Codec: {CODEC_NAMES[observation.codec[0]]}, '
              f'size: {observation.frame_size[0]}, '
              f'Bitrate: {observation.codec_bitrate[0]}, '
              f'QP: {observation.frame_qp[0]}, ')
        if done or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()
