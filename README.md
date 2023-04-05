# Intelligent Urban Traffic Signalization System using Multi-Agent Deep Deterministic Gradients.
<img src="https://img.freepik.com/free-vector/traffic-police-scene-with-stop-signal-flat-illustration_1284-61242.jpg?w=1800&t=st=1680685916~exp=1680686516~hmac=f331bee1e5315d3fc83df23d8b38fbc582b67b153d600b90f070f40881c022c5" style="margin:10px;width: 100%"/>

MADDPG (Multi-Agent Deep Deterministic Policy Gradient) is a reinforcement learning algorithm that can be used to train multiple agents to cooperate in a complex environment.

In the Poblacion area of Cagayan de Oro, an Intelligent Traffic Light System is being implemented to improve traffic flow and reduce congestion. This system uses the MADDPG (Multi-Agent Deep Deterministic Policy Gradient) algorithm to coordinate the actions of multiple traffic lights. By treating each traffic light as an agent, we can establish communication and information-sharing among them through a communication protocol. These agents can then learn to take actions that maximize a shared reward function, which includes metrics like average travel time, waiting time, and the number of vehicles passing through the intersection.

One of the critical metrics used as the reward function is the average waiting time per traffic light. The ultimate aim of this system is to reduce waiting times as much as possible, as this can significantly help in reducing congestion and improving traffic flow. As such, the agents are trained to take actions that result in the lowest possible waiting time across all traffic lights. This optimization of the metric can lead to a more efficient traffic management system, which can ultimately benefit the residents and visitors of Cagayan de Oro's Poblacion area.

This code is inspired from <a href="https://github.com/philtabor/Multi-Agent-Deep-Deterministic-Policy-Gradients">Phil Tabor's Implementaion</a> on <a href="https://github.com/openai/multiagent-particle-envs">Multi-Agent Deep Deterministic Gradients by OpenAI</a>.

The traffic environment's geographic setting was taken from OpenStreetMap and is modified to represent the concurrent traffic scheme as of 2023. The traffic data was collected through real-time on-site surveys done by students from Xavier University - Ateneo de Cagayan overseen by <a href="https://github.com/Nins16">@Nins</a>.

The main objective of this code is to develop an algorithm that utilizes Artifical Intelligence to thoroughly optimize the traffic light signalization in Cagayan de Oro's Poblacion area through the MADDPG Algorithm.


