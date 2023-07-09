Milestone
===

Track the development of new features

### 9th July
- [x] Issue of FEC. In pacing_controller.cc, the FEC packets are sent after all video packets are sent. The lost video packets will trigger retransmission via NACK before the arrival of the FEC packets. So, some of the FEC packets are not used before the completion of the rtx, which is a waste of bandwidth. 

### 8th July
- [ ] Figure out why the decoding may fail all the time
- [ ] Figure out why the decoding may success even if some RTP video packets are missing

### 7th July
- [x] Understand the logic of FEC encoding/decoding and log which packets are recovered by FEC.

### 3rd July
- [x] Many packets will be lost if the encoded image is much larger than the bitrate / FPS, causing high frame loss rate.
- [ ] The RTCP reported decoding/assmbly delay is a little larger than the real delay.

### 29th Jun
- [x] Find out the root cause of the decoding queue delay. It is not caused by the decoding delay. (It is caused by the VCMTiming)
- [x] Find out the causes of frame drop. (pacing delay, encoded size too large)

### 28th Jun
- [ ] NVDEC deocding does not return any frame on the first input. The reason is not clear. But it is not a big problem since we can anyway calculate the frame decoding delay. 

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
