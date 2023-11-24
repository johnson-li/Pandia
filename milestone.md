Milestone
===

Track the development of new features

### 24th Nov
- [x] Traing on the simulator with curricum level 0 is successful. The key trick is to set a small gamma (0.8) because an action affects a limite number of future rewards.

### 6th Oct
- [x] Improve the FPS of training the simple simulator

### 5th Oct
- [x] Implement a simple simulator for fast training, which will be utilized for transfer learning.

### 27th Sep
- [x] Implement the container env with a single program for both sender and receiver. It turns out that tc is quite reliable as long that we set the burst to the MTU, e.g., 1540 bytes.

### 12th Sep
- [x] The sb3 container implementation suffers from packet loss due to the use of UDP. Change to IPC socket.
- [x] The TC traffic shapping is observed of high RTT variance when multiple containers are running. Utilize the network emulation framework in WebRTC to solve this problem. WebRTC's simulation frameword is of high delay, abandoning it.

### 6th Sep
- [ ] Make the other control parameters constant to stablize the training.
- [x] Implement sb3 with container.

### 3rd Seq 
- [x] Migrate to docker and enable large-scale training.

### 27th Aug
- [x] Improve the training FPS.
- [x] TC-based network shaping is not working propertly. The buffer is too large. The problem is that netem causes abnormal delay when the buffer is built. So, we should avoid using netem and tbf together. The problem is also related to netns.
- [x] Reduce the CPU utilization of WebRTC (current value is 200%). The problem is solved by using inf. timeout in epoll_wait.

### 3rd Aug
- [ ] Solve the issue of the zig-zag pattern of the rewards.

### 2nd Aug
- [x] Using a continual reward function makes it possible to train the model to react to the observation on the delay. However, on problem is that the reward has a zig-zag pattern, moving between a high value and a low value. One reason is that, at some point, the pacing rate drops to a low value. The pattern looks like: low delay -> high bitrate (action) -> low pacing rate (unclear reason) -> high delay -> low bitrate (action) -> low delay -> [inf.]

### 24th July
- [x] It is hard to teach the model of choosing a low bitrate when overshooting because it takes a lot of steps to get recovered even if the bitrate is set to a low value. The correct action (low bitrate in this action) is hard to be sampled during training because it is by default ramdomized. The solution may be 1) making the training process hybrid, 2) offline training with GCC at first (imitation learning), 3) making the reward more smooth relative to unexpected conditions, instead of simply using penalty 4) ?. The root cause if the inefficiency of exploration. Regarding this, we are looking for a better context-aware exploration strategy than vanilla possibility-based random action sampling used in DRL.

### 23th July
- [ ] The training performance of rllib is really bad. Try with sb3. The performance issue exists in rllib, too. The reason may be with the performance guantee module in webrtc (e.g., the frame dropper). We may need to fix the action values if they are not to be trained.

### 21th July
- [x] The training may be very bad if the bitrate set by DRL is further adjusted by webrtc. (Have applied the bitrate just before encoding. But the training is still bad.)

### 20th July
- [ ] When the decoding equeue is built, it never recovers (or takes a long time). The problem occurs when fps = 60.
- [x] It takes a long time for the pacing queue to recover, which makes the following training steps meaningless. A temporal solution is to terminate the training when the pacing queue is not cleared within a timeout.
- [x] The training may be very bad if the observation is of high dimension. (The training is still bad after reducing the dimension of the observation.)

### 15th July
- [x] Refactor the env (observation, monitor block, action, etc.)

### 14th July
- [x] Log the cause of frame drop and downscale (The first few frames are dropped due to bitrate constraint, which trigers downscale. It takes a long time to recover from downscale.)
- [x] Find out the root cause of the pacing queue, which builds up very long pacing delay. The pacing rate has been increase to a very large value. (The problem is that the actual egress rate is much smaller than the pacing rate (7 mbps vs 12 mbps). Since the pacing rate and the bitrate are large enough for the encoded frame, the frame dropper does not take effect. The egress queue just builds up.) [The egress rate measured per second is limited to 7 mbps. Linux stdout is blocking. The python program delays the webrtc process because the data is not processed fast enough.]

### 12th July
- [x] Transfer frame-related metrics along with the packet RTCP feedback. (it is over dedicated rtcp packet)

### 10th July
- [x] The first RTP packet (RTP id = 2) may be dropped, causing H264 decoding failure. (never happen again)
- [x] For some reason, the first encoded frame from NVENC may be dropped by the decoder. (not a big issue. it also happens in OpenH264)
- [x] NVENC has two key frames at first. (OpenH264 is the same) 
- [x] The pacing rate of the first frame is much lower than the value set by DRL. (It is because probing is in effect, which is controlled by GCC)

### 9th July
- [x] Issue of FEC. In pacing_controller.cc, the FEC packets are sent after all video packets are sent. The lost video packets will trigger retransmission via NACK before the arrival of the FEC packets. So, some of the FEC packets are not used before the completion of the rtx, which is a waste of bandwidth. 

### 8th July
- [x] Figure out why the decoding may fail all the time. (Issue 1: delayed encoding return [solved]. Issue 2: SPS is missing in the second IDR frame)
- [x] Figure out why the decoding may success even if some RTP video packets are missing (It does not happen. The wrong observation is caused by the bug in the analyzer)

### 7th July
- [x] Understand the logic of FEC encoding/decoding and log which packets are recovered by FEC.

### 3rd July
- [x] Many packets will be lost if the encoded image is much larger than the bitrate / FPS, causing high frame loss rate.
- [x] The RTCP reported decoding/assmbly delay is a little larger than the delay caused by the RTT. (The problem is solved by reporting the UTC time)

### 29th Jun
- [x] Find out the root cause of the decoding queue delay. It is not caused by the decoding delay. (It is caused by the VCMTiming)
- [x] Find out the causes of frame drop. (pacing delay, encoded size too large)

### 28th Jun
- [x] NVDEC deocding retruns after 3 API calls. It causes issue because the required key frame is generated under delay. So, the higher layer may get an outdated delta frame even if it is requesting a key frame, causing decoding failure. The best solution is to ensure that the API call returns imediately, instead of delayed. A workaround is to detect the delayed key frame in the higher layer, discarding the outdated delta frames. (m_nExtraOutputDelay in NvEncoder.h)

### 21th Jun
- [x] Support nvenc/nvdec

### 20th Jun
- [x] Add ntp client and server 

### 18th Jun
- [x] Support RTCP RTT estimation
- [x] Implement OnRL observation (loss, delay, delay interval, throughput, and gap)
- [x] Implement OnRL reward
- [ ] Implement the safety condition detector as in OnRL
- [ ] Implement the hybrid mode, i.e., RL + GCC, as in OnRL
- [ ] Support federated learning (learning aggregation), as in OnRL

### 15th Jun
- [x] RTCP report frame deocding start time
- [x] Support qlen in traffic control so that we can simulate the router buffer size

### 14th Jun
- [ ] Add script of generating replay buffer from logs for offline training
- [ ] Support offline training
- [x] Add evaluation code
- [x] Support non-linear normalization for values of large range (SAC/SAC_None_e4c6c_00000_0_2023-06-14_15-13-38, the action stucks at low-bitrate, resulting in low reward)

### 12th Jun
- [x] Add resolution change penalty in the reward to keep the resolution stable. (SAC/SAC_None_48fb1_00000_0_2023-06-13_10-09-49, the result is close to SAC/SAC_None_6fc5b_00000_0_2023-06-11_08-47-22)

### 11th Jun
- [x] Use (resolution, bitrate, pacing rate) as the action. (SAC/SAC_None_6fc5b_00000_0_2023-06-11_08-47-22) (the reward is much worse than a fixed resolution)

### 9th Jun
- [x] Migrate WebRTC env to rllib external agent. (~/ray_results/SAC/SAC_None_d24ac_00000_0_2023-06-10_15-03-33)
- [x] Train the first model with only one action, i.e., bitrate, on a fixed bandwidth, i.e., 1 gbps. (zoo SAC_13)
- [x] Extend the first model with changable bandwidth during training.
- [x] Extend the first model with a second action, i.e., pacing rate. (SAC/SAC_None_2df5f_00000_0_2023-06-10_22-08-27)
