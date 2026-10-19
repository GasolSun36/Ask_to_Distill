"""tau2 agent domain: synthesizer RL, card harvest, OPD with a multi-turn agent loop, evaluation.

Two interpreters are involved. ``a2d`` always runs in the a2d env (verl, vLLM,
torch); tau2 itself and its dependency chain (loguru, litellm, ...) live in a
separate tau2 env that has neither torch nor pyarrow. Code is split accordingly:

* ``envside/`` runs only under ``$A2D_TAU2_PYTHON`` and imports only the
  standard library, tau2, fastapi/uvicorn/requests/openai and other envside
  modules. It is started as ``python -m ask2distill.tau2.envside.<module>`` by
  ``ask2distill.tau2.spawn`` and takes its settings as rendered ``TAU2_*``
  environment variables plus command-line arguments.
* ``trainside/`` runs in the a2d env inside verl (reward worker, agent loop,
  hydra entry, launchers) and never imports tau2; it reaches the environment
  over HTTP (``envside.env_service``).
* The modules directly in this package (config rendering, spawning, serving,
  banks, selection, summaries) run in the a2d env.

This file must stay import-free: the package is imported by the tau2
interpreter too.
"""
