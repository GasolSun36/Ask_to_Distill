# Reproducing the main table

The main table compares, for each teacher-student pair, OPD on the synthesizer's question bank
(**Ours**) with the same OPD recipe on real questions (**Vanilla**). In the commands below, `<N>`
is the OPD checkpoint step to evaluate.

There are two levels:

1. **Evaluate the released checkpoints** (GPU hours): the Ours student of each pair is the HF
   model `gasolsun/<pair>`; `a2d eval math --model gasolsun/<pair>` (or `eval code`) downloads it
   to `$A2D_MODELS` and evaluates it, with no merge (README, "Evaluation").
2. **Retrain the OPD arms on the released banks** (8 GPUs per arm, several 4 h windows each),
   then evaluate. Re-running stages 0-3 (calibration, synthesizer RL, selection, harvest, filter,
   bank) is possible for every pair with the commands of docs/PIPELINE.md, but needs the full
   synthesizer-RL compute.

Commands below use the SLURM wrappers; drop `launchers/slurm/submit.sh <stage> <pair>` in favour of
`a2d <command> --pair <pair>` to run inside an existing allocation.

Experiment names are `opd_<pair>_ours` and `opd_<pair>_vanilla` (`opd.arms.<arm>.experiment`);
results land in `<experiment>_step<N>`.

## Math (avg6 = mean of AIME24, AIME25, AIME26, AMC23, BRUMO25, MATH500)

```bash
P=Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500; ARM=ours; STEP=<N>
launchers/slurm/submit.sh opd $P --arm $ARM     # with launchers/slurm/resubmit.sh --target <total steps>
for T in AIME24 AIME25 AIME26 AMC23 BRUMO25 MATH500; do
    launchers/slurm/submit.sh eval-math $P --arm $ARM --step $STEP --tasks $T
done
a2d collect math --runs $A2D_OUT/eval/math/opd_${P}_${ARM}_step$STEP
```

Total OPD steps: `Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500` Ours 90 (2,889 questions, 2 epochs), Vanilla 542;
`Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B` Ours 400; `Math_DeepSeek-R1-Distill-Qwen-7B_Skywork-OR1-Math-7B` Ours 400, Vanilla 271.
Evaluation samples 16 answers per problem (MATH500: 4) at temperature 0.7, top_p 0.95,
31,744 new tokens, thinking disabled. vLLM sampling is not seeded per request
(docs/DEVIATIONS.md, quirk 13), so a re-evaluation varies within sampling noise.

## Code (Avg = mean of LCB v5_v6 pass@1, MBPP+ lenient, APPS introductory / interview1k / competition, CodeContests)

Avg follows the paper table: every score is rounded to one decimal as its cell prints, the six
cells are averaged (the cell mean) and the Avg is that mean rounded half-up to one decimal.
`a2d collect code` prints `avg`, `avg_cells` and `avg_exact` (the mean of the unrounded scores).

The Ours students are published as `gasolsun/<pair>`.

```bash
P=Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507
launchers/slurm/submit.sh opd $P --arm ours                       # 156 steps
launchers/slurm/submit.sh eval-code $P --arm ours --step <N>      # LCB n=4 + EvalPlus + extra benches
a2d collect code --runs $A2D_OUT/eval/code/opd_${P}_ours_step<N>
```

LCB runs 342 problems with n=4 at temperature 0.7, top_p 0.8, top_k 20 and 8,192 tokens; MBPP+
is greedy with 1,024 tokens and is rescored with the lenient entry-point repair (the reported
number); the extra benches use n=1, temperature 0.2, 4,096 tokens. HumanEval+ is reported but not
in the Avg. The result directory is always `<experiment>_step<N>`: LiveCodeBench reuses
generations by tag, so a tag shared between runs silently reuses the wrong outputs; the code
refuses anything else.

## tau2 (pass^1 per domain on the official test split; mean of the three)

Pair `Agent_Qwen3-4B_Qwen3-4B-Thinking-2507`; the served name of an arm checkpoint is
`<pair>-<arm>-s<step>` (`eval.served_name`; the "Qwen" in it routes the model in ToolOrchestra).

```bash
P=Agent_Qwen3-4B_Qwen3-4B-Thinking-2507
launchers/slurm/submit.sh tau2-opd $P --arm ours                   # 2 nodes: trainer + user simulator
# compute node: merge with the student template and serve; login node: the eval client
a2d merge --pair $P --arm ours --step <N> --chat-template student
a2d tau2 serve --pair $P --plan eval --arm ours --step <N> --out $A2D_OUT/tau2/eval_endpoints.txt
a2d tau2 eval --pair $P --endpoints $A2D_OUT/tau2/eval_endpoints.txt --tag test4
a2d tau2 summarize --pair $P --models $P-ours-s<N>
```

Evaluation uses 4 trials per task, OpenAI `gpt-4o-mini` as the user simulator (your key; see
docs/TAU2.md), up to 200 steps per dialogue. Telecom is in neither training set.

## Expected compute

| Stage | Hardware | Time (final runs) |
|---|---|---|
| Synthesizer RL, math / code | 16 GPUs | per step: `Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500` 172 s, `Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B` ~333, `Math_DeepSeek-R1-Distill-Qwen-7B_Skywork-OR1-Math-7B` 517-591, `Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507` 348-405, `Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct` 103, `Code_Qwen3-4B_Qwen3-4B-Instruct-2507` 449-488; 60 steps |
| OPD arm | 8 GPUs | 90-542 steps of 64 prompts x 4 rollouts, over several 4 h windows (resumed by `resubmit.sh`) |
| Math eval, one task | 1 GPU | 16 x 30 problems x up to 31,744 tokens |
| Code eval, one point | 2 GPUs | LCB dominates; EvalPlus runs problem by problem |
| tau2 synthesizer RL, one version | 3 nodes | 60 steps |
