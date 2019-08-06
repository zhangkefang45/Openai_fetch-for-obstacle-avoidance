﻿# Openai_fetch-for-obstacle-avoidance
# 使用openai_gym中的fetch结合RL算法DDPG来进行避障，在最初，我们使用规划末端的方式来进行reach，进展顺利，但是在进行关节的避障时，发现并不能很好的进行，我们打算采取以下的顺序来达到最终的目的。
0. 到达空间，使得关节操控的机械臂到达某个固定的区域，选择范围更加的广泛，理论上更容易到达。（选）
- [x] 1. 使用关节来操控fetch的机械臂，为了简化，首先使用三个关节来对应规划末端的三维空间达到reach的效果（如有必要则减少至两个关节)
- [ ] 2. 在1的基础上进阶的进行一些操作：  
 > - [x] 2.1 为了使关节更加灵活，当三个关节成功时，解锁更多关节，使用四个或五个关节达到上述效果  
 > - [ ] 2.2 使用一些技巧，例如HER来使得三个关节更好的达到目标  
p.s. 2.1和2.2并行处理，能够同时达到最好，则使用其一来进行接下来的操作
- [x] 3. 使用上述操作测试带有障碍检测的reward，观察是否能够训练成功
- [ ] 4. 关节确定能够进行reach操作时，希望进行一些避障，加入障碍物从新从上述过程遍历。
> - [x] 4.1 加入悬空小障碍物训练随机目标
