# Sense
Scheduling structural learning

Objective:
  + Decision on the type of structural learning:
    + [ADMM](http://openaccess.thecvf.com/content_ECCV_2018/papers/Tianyun_Zhang_A_Systematic_DNN_ECCV_2018_paper.pdf)
    + [AdaNet](https://arxiv.org/pdf/1607.01097.pdf)
    + [DARTS](https://arxiv.org/pdf/1806.09055.pdf)
  + Learn Flux.jl:
    + [Julia Academy](https://juliaacademy.com/p/introduction-to-machine-learning)
  + Implement the structural learning


Project Checkpoint Summary:
In this phase of the **SENSE** (**S**ch**E**duli**N**g **S**tructual l**E**arning), the [DARTS](https://arxiv.org/abs/1806.09055) algorithm was applied to DQN agents in CartPole. To our knowledge, DARTS has not been applied to Reinforcement Learning or implemented in Julia before this project.

[CartPole](https://github.com/openai/gym/wiki/CartPole-v0) is a relative simple task, with four observable enivronment variables and two agent control variables. Networks frequently used for DQN agents to solve Cartpole are usually fully connected networks with two hidden layers. 

To apply DARTS to CartPole, the network architectures were initially searched completely in parallel. Each operation was a full network. 

In all following experiments, both the first and second order approximations were used to compare the efficiency and effectiveness of each. The second order approximation is significantly harder to implement and slower to train, due to the multiple forward and backward passes required per step. 

In the following figure, six different dense network architectures were used as operations. The hidden dimensions are listed in the legend. The first five architectures in each training regime (first order approximation and second order approximation) had two hidden layers, while the final one had a single hidden layer. The softmax weight of each architecture is plotted over the course of training. In both training regimes, both the architecture with two hidden layers each of size 128 as well as the architecture with a single hidden layer of size 1024 initially increased in architecture weight, while the other four smaller networks decreased. However, later in training, the 1024 architecture fell back down while the 128x128 architecture continued to dominate. It is generally preferable in DARTS for a single operation to dominate, since that operation would be used for the sole operation for that portion of the macroarchitecture after the algorithm concludes. 
![Initial dense network experiment](https://uc2e19bbfc5576e96e3a46a250e0.previews.dropboxusercontent.com/p/thumb/AAzrxv4NKChlSNzZadKX8NPgwJGn44JMF5bBlDeG3341-TnQosxvbEeQcMERKEOrBkwCDTsfb8AMjmXZf-WiMh4ZH75tEhMkhISf_vIIX0ED8U2i2zJiJZIA8NVH-Fiqi3_qO3ZDzAcZVEyiM2SRRu6nT886H3a1BTZ_lMAeC1OzqZW64Q2g9jdBSaxPyHRPel4hJ5rqrppVA-AQir0SHkD8_W3Zwts_3t13B2vnMqdSMPvTSfhfNlwV3-xxIMbEBbA-NFripoDMFfQPbWv6Zq9d5YJBYDjri31KU1DhHswrMNYnXmQZJMlIhnGAsG81KItdzY-y7yISUfxtWelFyxDjgtR1HVRve_v279rtWJIDcZPcvry7BEsPdlR3mn6OGx4/p.png)


Since the 128x128 architecture was very successful and is very commonly used in DQN networks for CartPole, multiple very similar architectures were tested as operations. In the following figures, three networks each with double hidden dimensions of 125, 126, and 127 were used as the operations, and the training was run much past the initial "convergence". The architecture weights are plotted first, and the reward per episode is plotted second. Both training regimes initally approached the same architecture weights with 126x126 having the highest weight, but over time after this convergence, the first order architecture weights stayed relatively staic while the second order architecture weights slowly changed. In particular, the architectur weight for 127x127 eventually surpassed that of 126x126, thus changing the architecture found via DARTS.
![Similar dense networks - architecture weights](https://uc2085d94cb98981264e83150905.previews.dropboxusercontent.com/p/thumb/AAwO5BfJkSMSZMFYd8YxHlHuTXb9DjtPyO0XdWgEn0OIFpJ4ryZe4gJwPnMQ0NgFydvZKUEwKcNg_sO7MgSEeWf8iWaDUdOp_2JGBWVAQcXKm0EnByOAVBQ7mrD5YgerSKq3oqBwoipQvhrBDIqPX4inyGmjIRbZ8gNTFV1dQvptInAtlH7nx_qsI_w7cENhLlCLNbJ0NrD2Uc9J-zcmwnBdE8TtfzbWg2cW11n7HYnEg4Mm4eqjp3EUWMuJfUQsiRJuLp8zgCu8Vimuopjn2XaHSt9vdNGq-5BV45Y3UsQ0XtYr86-wDEcSfUbbVAxVQmVlll-EoU7DL6HEH8z3NWfdniEA3JHWjsVRFzp1b_kw9I4eeM8vjRu5F-wyQYxYGts/p.png)
![Similar dense networks - rewards](https://uc191b376dbd229c89eb1359919a.previews.dropboxusercontent.com/p/thumb/AAypuTNdEnM2D0Lreq-uHQVLe7_QRybUDhfoiKK6gAFsFsvG-ESp_0mafmmLrOovWTPQcT9xZg49TGL6Qw_FcQ0ZOnJfQakdd4IYeAByAiZTg1U7UGEusJ1qHtfTcZYlHY3ZBa2sqg0_dgzUcOLuizrBeffdyPUYnkyN-WwpweI3gWCquA4NJVNVTPuC62nnmairB64U6xw7g9pFkcyvBURo4-k1zC0cIJEkt-YzYOgSe6E2Ojt8nX-CjSJNqVjnlL9-hSAjJTSbNB2qs--EQk2zw5BA9fJUNyPfXs6-6tiQiQ-fsrZwA0QO9mOUTMOA-GWgTf4-3a0rIJXd1kefZ_lum81EM_lg7BsX032oHPpDaEt67GTepPoH5-3jojgaBEw/p.png)

In the next set of experiments, small convolutional network architectures were used as parallel operations. In the first experiment, three different architectures were searched over, and the regimes showed similar architecture weights but very different rewards per episode. 
![Three parallel convolutional operations - architecture weights](https://ucc10dfef5230a25483a13c8b079.previews.dropboxusercontent.com/p/thumb/AAwVI_L0rzS6WMUeVgOUC9AaAvbJeaDfbFaX9ptlTDirpVmXciaJBPWMMpg6zkGFETDeQ5__B56wetBRzHEBSqQfzm1POJPHoE9Nu76ZjCXDxXVMsu-eFkfyfA3lQL-SQONvxFTAgnCVfxakP4Q9-abbQkuvpSbr-iX74DsTXStA6ulDfv3GgMubwpJbILfhdKm81zmeVce-kBJvBJaUdnpG_NNwQhpcdruS_Jef1PCRpvHTN-AklSLzvEpYr4_0oEzTPmmi6kVMU8dH6i1wg6GZhgqbeTbskdTnSV_ZyhvnJLlsYZEi1gI6z1kB4mO17NC3SHOp6gfQeCVLYSYsFHZFzLfjPigl745GKqMsqnHHhWlhKm-zcXvG3NqFcFsv5_4/p.png)
![Three parallel convolutional operations - reward](https://uc69181d486e13c09447f9d628a8.previews.dropboxusercontent.com/p/thumb/AAzcWPS3mE6qy0dQj6uQRMbmsyZvr78tZuAy8NnoOaZ2UBJoFWP2n4ksgNGZwU8Rabj2fl8E8ZZ1Ozr5ooBK9M8VM85yJrWO4F4RPOgXCu-Uju5CR_34_N3LORx7PB-nqMefFFpBQWOMkmnCeuAdR7GGohL0p2iciW-vg46ZPIkCik64m9zFvOZd7Lc7gkPWIswNEawuwLQPmyFT9AGm9YXOKeZArEOAoqz6E2_WwyIQVauY0bhocIenEXMNHWtfJBIJ-6dwhEkJ83ryPIWAcF8fbGiZ0nY1uh-vFZnRvT_JC9ao7stMfF2fu2QsF0O5nuglYs2YOeI2vL88YhmMA48ysbCh1g04YsDiUEqy4S24E25hK2DL3pCUS7BMaffGcDw/p.png)

When a fourth operation was added, the training is much less successful. This may be due to bad initialization.
![Four parallel convolutional operations - architecture weights](https://ucb9aa21d16f10957f67b47178d4.previews.dropboxusercontent.com/p/thumb/AAx2cQQT9nUKL0z58riMChMf4FxKgF2Yc3PoiMjBPQeBzSN0dtCMhAHLKSqieWQLKqY-3en8tcBicdFuQ4cj2vcRqM8uKttDA49ypCPTvLwHGKxtv0SStodcTxcgjPgqULUh3y5Ek702m2OV0MQNXSclYfuhqDm_RVLupRKvfeo7kWjM8Xr5iHPszU3NRfFaEPnzpDR5yVUPkWjS0QuVurIo2XlEGo1efVd_8PMWrpmcpmNs74k3-1MNMEsC8fLfc16ip_vFSbKqBVWAy1GC-yB1L08XBQMVMez64s4g6zY1AtyFVogv8mnnnfrno9P9MJGSRGt7A7whIC95UlJVzFDBhevEzh2Osa8X6Qjrzuai26XYNzWLkvl3YCANRrvQyjk/p.png)
![Four parallel convolutional operations - reward](https://ucd9d88d0bf46951563b2e6eee78.previews.dropboxusercontent.com/p/thumb/AAxvL3W9nb6TGFSfw9D8sGeffjddvfK9tspQrZNBgN_c_QNsAq4-0mFl0iVCHWRpWFOSS4r5M4TV1YApxv2zIJ_SmU2OkoqCMFskpcd0gqiAVowL2sojDugUn_-_8dLfRvam--j2X2i2dmhGCdudd-_A-aAuRlM0zdvAQrZee_8Gx4Dc9svTEywAwz3pH0GV5TDE1RRgwk1Aqf7k7Jv_elRftKxkRtqZk_FZVGsaSMjWM9kIv-ZYMuR2x_4fn3uMfSIqBCS1llLyKsreD2gMZbHFq9dboMTlJo2TretA6lobaZvuwSee1zZ6Z7LTIrQNABWUGxHq4YIBYeyqTMTktFP3774-VN0XMJnRPMeHtL-XSZSXGgAUtM0QByzY_Ur0hag/p.png)

Now that convolutions are implemented, individual layers can be used as operations. 


In general, the main difference between the first order approximation and second order approximation is training speed. Both training regimes typically approached similar architecture weights, but the second order often approached them slightly faster. Also, the second order approximaiton was generally less stable in the architecture weights, although future experiments could investigate whether this instability leads to eventual convergence or indefinite oscillation.
