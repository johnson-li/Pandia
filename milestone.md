Milestone
===

Track the development of new features

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
- [ ] The RTCP reported decoding/assmbly delay is a little larger than the delay caused by the RTT.

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
- [ ] Support qlen in traffic control so that we can simulate the router buffer size

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
