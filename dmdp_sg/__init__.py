"""Practical DMDP implementation and its evaluation on the Safety-Gymnasium HalfCheetah velocity task.

Code for Section 4.5, Section 5, and Appendices E and F of "Dissimilarity-Based Markov Decision Processes for
Stabilizing State Distributions".

Modules
-------
config    experiment settings, target distribution rho_star, methods          (Section 4.5, Appendix E)
targets   target families of rho_star, the exact W2 of a sample to rho_star, and the threshold D_thr (Appendix E)
envs      cost penalty (pilot only), sensor noise, window estimate omega_hat_t, dissimilarity D, threshold reward r_tD  (Section 4.5, Appendix E)
agent     actor-critic network of the PPO learner, with reward and cost critics  (Appendix E)
train     PPO-Lagrangian learner with the shaped reward r_t + r_tD - beta * D    (Section 4.5, Appendix E)
sac       SAC and SAC-Lagrangian learner, the second learner of the study       (Appendix E, Section 5.1)
evaluate  evaluation rollouts at fixed sensor-noise levels                    (Section 5)
metrics   stability metric D(rho_k || rho_hat), its stationary reference value, distance to rho_star (Section 5, Appendix F)
analysis  loading and aggregation of the stored results                       (Section 5, Appendix F)
"""
