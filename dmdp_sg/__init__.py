"""Practical DMDP implementation and its evaluation on the Safety-Gymnasium HalfCheetah velocity task.

Code for Sections 6-7 and Appendix D of "Dissimilarity-Based Markov Decision Processes for
Stabilizing State Distributions".

Modules
-------
config    experiment settings, target distribution rho_star, methods          (Section 7.1, App. D)
envs      sensor noise, sliding-window estimate omega_hat_t, dissimilarity D  (Sections 6.1-6.3)
agent     actor-critic network with reward and cost critics                   (Appendix D)
train     PPO-Lagrangian training with the shaped reward r_t - beta * D        (Section 6.3)
evaluate  evaluation rollouts at fixed sensor-noise levels                    (Section 7.1)
metrics   stability metric D(rho_k || rho_hat) and its sampling floor         (Section 7.1, App. D)
analysis  loading and aggregation of the stored results                       (Section 7)
"""
