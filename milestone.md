Milestone
===

Track the development of new features

### 14th Jun
- [ ] Add script of generating replay buffer from logs for offline training
- [ ] Support offline training
- [ ] Add evaluation code
- [x] Support non-linear normalization for values of large range

### 12th Jun
- [x] Add resolution change penalty in the reward to keep the resolution stable. (SAC/SAC_None_48fb1_00000_0_2023-06-13_10-09-49, the result is close to SAC/SAC_None_6fc5b_00000_0_2023-06-11_08-47-22)

### 11th Jun
- [x] Use (resolution, bitrate, pacing rate) as the action. (SAC/SAC_None_6fc5b_00000_0_2023-06-11_08-47-22) (the reward is much worse than a fixed resolution)

### 9th Jun
- [x] Migrate WebRTC env to rllib external agent. (~/ray_results/SAC/SAC_None_d24ac_00000_0_2023-06-10_15-03-33)
- [x] Train the first model with only one action, i.e., bitrate, on a fixed bandwidth, i.e., 1 gbps. (zoo SAC_13)
- [ ] Extend the first model with changable bandwidth during training.
- [x] Extend the first model with a second action, i.e., pacing rate. (SAC/SAC_None_2df5f_00000_0_2023-06-10_22-08-27)