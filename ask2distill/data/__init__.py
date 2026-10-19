"""Models, datasets and the data release: download, build, verify.

* ``models``                pinned base-model downloads (HF or the ModelScope
                            fallback) and the ``*-SynNoThink`` synthesizer bases;
* ``download``              fetch the data release into ``$A2D_DATA``;
* ``manifest``              the committed ``data/MANIFEST.json`` and its checks;
* ``build_math_data``       DAPO-Math-17k and the math test sets;
* ``build_code_eval_data``  LiveCodeBench / EvalPlus caches, APPS and CodeContests;
* ``build_public``          OpenCodeReasoning at a pinned revision and the
                            OpenCodeReasoning-10k Vanilla code arm;
* ``tau2_data``             the tau2 domain data, official split and tokenizer cache.
"""
