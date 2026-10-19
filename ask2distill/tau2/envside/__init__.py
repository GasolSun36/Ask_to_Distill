"""Modules that run under the tau2 interpreter (``$A2D_TAU2_PYTHON``).

Never import ``ask2distill.config``, omegaconf, torch or pyarrow from here:
the tau2 env does not have them. Settings arrive as environment variables
rendered by ``ask2distill.tau2.env`` and passed by ``ask2distill.tau2.spawn``.
"""
