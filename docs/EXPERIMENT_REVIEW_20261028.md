# Experimental protocol review 2026 10 28

The release was checked against Sections 4 and 5 of the supplied Word manuscript,
SGSA_Static Graph with Statistical Attention_Word2.docx. This review changes source
validation and checkpoint handling; it does not replace reported results or claim
that the full training experiments were repeated.

## Settings retained from the manuscript

SDWPF uses 134 turbines, 16 features, 20,000 time steps, a 96-step history and a
48-step horizon. The chronological split is training [0,14000), validation
[14000,15000), unused [15000,16000), and evaluation [16000,20000). The strides are
4 and 24, with 3,465 training, 40 validation and 165 candidate evaluation windows.
The reported valid evaluation count is 966,746 prediction points over 153 windows.
Checkpoint selection uses validation MSE, learning rate 0.0005, at most 30 epochs
and patience 5.

Hill of Towie uses 21 turbines, 7 features and a 96-to-48-step forecast, with
729/26/26 windows and 26,187 valid evaluation prediction points. Checkpoint
selection uses validation MAE, learning rate 0.001, at most 50 epochs and patience 8.
Both datasets use rho 0.25, dropout 0.1, batch size 32 and seeds 2024/2025/2026.
Metrics remain on the training-statistics z-score scale. Baseline adaptations
remain documented, including the structural adaptation of MultiGSA.

The original SDWPF batch order is retained: changing it would change training
trajectories and would require new results. The two efficiency protocols retain
their separate warm-up/repetition counts of 20/100 and 10/30.

## Repairs

- Run reuse now checks the saved configuration, including seed, strategy, split,
  coefficients and training settings. SDWPF also checks its microbatch and shuffle
  choice before accepting completed runs or resuming a checkpoint.
- New checkpoints record rho in hash-bound metadata; loading rejects a different
  rho even when all tensor names and shapes match. Legacy weights remain loadable,
  but their rho cannot be established from a plain state dictionary. The standalone
  SDWPF evaluator records whether rho was verified from checkpoint metadata.
- The standalone SDWPF trainer uses the documented precision flags and checkpoint
  writer. Evaluation rejects nonfinite predictions; the standalone SDWPF evaluator
  also checks the manuscript's valid prediction count. Hill21 training rejects an
  epoch with no valid labels.
- The unused clean split helper now requires a complete history at training start
  and produces the same forecast origins as the benchmark. This does not change
  the benchmark's existing split or its reported results.
- Hill21 now checks its frozen settings, including architectural values that its
  adapters previously fixed internally. SDWPF also checks clipping, selection,
  metric scale and corruption seeds against the frozen protocol.
- Structural checks validate output shape and finiteness as well as parameter
  counts. Missing core dependencies and assertion failures can no longer be
  reported as successful optional skips. Window-count equality is checked.
- Requesting help from the hash-manifest utility no longer rewrites the manifest.

## Manuscript limitation that code edits cannot remove

The supplied experimental sections explain chronological splitting, training-only
preprocessing, standardized metrics, three-seed variability and baseline
adaptations. They do not state the prior developmental inspection of the
evaluation panels documented in this repository. That disclosure should also
appear in the manuscript's experimental limitations. In particular, rho was
compared at 0.5 and 0.25 on the SDWPF development evaluation panel; it must not be
described as chosen solely on validation data or fixed before all panel inspection.
Hill21 learning-rate/dropout selection used validation MAE, while its evaluation
period had also been inspected during development.

The disclosure is retained in the configs, README and development history. Removing
it would not create an untouched confirmation set. A new independent evaluation
would need separately justified data and new experiments.

The original Word file and all reported result tables are unchanged by this source
review. See provenance/smoke_checks.json for current structural checks and
tools/test_integrity_guards.py for the CPU regression cases. Those checks do not
verify the paper's accuracy tables or reproduce the GPU timing measurements.
