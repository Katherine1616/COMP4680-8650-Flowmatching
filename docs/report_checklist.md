# Report Checklist

Use this file as a running checklist when writing the final assignment report.

## General Deliverables

- Document the implementation approach and experiment setup for each part.
- Include all requested visualizations and results.
- Describe observations in your own words.
- Answer every written question using your experiment results.

## Part 1: Warm-up

### 1.1 Data Visualization

Report:

- Six scatter plots total.
- For each dataset, include:
  - Original `D=2` data.
  - `D=32` data projected back to 2D with `to_2d()`.
- Datasets:
  - `swiss_roll`
  - `gaussians`
  - `circles`
- Briefly state that the back-projected `D=32` data preserves the same visible structure as the original `D=2` data.

Current output files:

- `outputs/part1/data_swiss_roll_d2.png`
- `outputs/part1/data_swiss_roll_d32_back_projected.png`
- `outputs/part1/data_gaussians_d2.png`
- `outputs/part1/data_gaussians_d32_back_projected.png`
- `outputs/part1/data_circles_d2.png`
- `outputs/part1/data_circles_d32_back_projected.png`

### 1.2 v-Prediction Flow Matching at D=2

Report:

- Hyperparameters used for flow matching training and sampling.
- State that this uses `v`-prediction with `v`-loss:
  - Forward process: `z_t = (1 - t) x + t eps`
  - Target velocity: `v = eps - x`
  - Loss: `MSE(model(z_t, t), eps - x)`
- Include generated sample scatter plots for all three datasets at `D=2`, each alongside ground truth.
- Describe whether the generated samples resemble the training distributions.

Current hyperparameters:

- Model: 5 hidden layer MLP, 256 hidden units, 128-dimensional sinusoidal time embedding.
- Optimizer: Adam.
- Learning rate: `1e-3`.
- Batch size: `1024`.
- Training steps: `25,000`.
- Sampling: Euler ODE from `t=1` to `t=0`.
- Sampling steps: `50`.
- Number of samples plotted: `4096`.
- Device used for current run: CPU.

Current output files:

- `outputs/part1/generated_swiss_roll_v_pred_v_loss_d2.png`
- `outputs/part1/generated_gaussians_v_pred_v_loss_d2.png`
- `outputs/part1/generated_circles_v_pred_v_loss_d2.png`
- `outputs/part1/part1_config.json`

## Part 2: Flow Matching Parameterization

### 2.1 Derive x/v Conversion Formulas

Report:

- Derive conversions between `x`-prediction and `v`-prediction from:
  - `z_t = (1 - t) x + t eps`
  - `v = eps - x`
- Useful derivation:
  - `eps = x + v`
  - `z_t = (1 - t)x + t(x + v) = x + t v`
  - Therefore `x = z_t - t v`
  - Therefore `v = (z_t - x) / t`
- Note numerical care near `t=0` if converting from predicted `x` to `v`.

Implementation note for this assignment:

- The assignment uses `t=0` for clean data and `t=1` for noise.
- This is the reverse of the notation used in the JiT paper, which writes the linear interpolation with clean data at the other endpoint.
- In the report, use the assignment convention consistently.
- For `x`-prediction:
  - Network directly outputs `x_theta`.
  - Convert to velocity with `v_theta = (z_t - x_theta) / t`.
- For `v`-prediction:
  - Network directly outputs `v_theta`.
  - Convert to clean data with `x_theta = z_t - t v_theta`.
- The implementation clips training times to `[epsilon, 1 - epsilon]` for numerical stability. Current Part 2 default: `epsilon = 1e-2`.

### 2.2 Reproduce Paper Finding

Report:

- Implement and compare all 4 prediction/loss combinations:
  - `x` prediction + `x` loss
  - `x` prediction + `v` loss
  - `v` prediction + `x` loss
  - `v` prediction + `v` loss
- Run each combination across:
  - Datasets: `swiss_roll`, `gaussians`, `circles`
  - Dimensions: `D=2`, `D=8`, `D=32`
- Total experiments: `4 * 3 * 3 = 36`.
- Include generated sample visualizations for every configuration.
- For higher-dimensional samples, visualize by projecting generated samples back to 2D with the dataset projection matrix / `to_2d()`.

Current implementation files:

- `src/flow_matching.py`
  - `train_flow_matching(...)` supports all 4 `x/v` prediction-loss combinations.
  - `sample_euler(...)` supports both `x`-prediction and `v`-prediction sampling.
- `scripts/part2.py`
  - Runs the full 36-experiment grid by default.
  - Saves one generated-vs-ground-truth figure per configuration.
  - Saves checkpoints and a JSON result record.

Default Part 2 command:

```bash
.venv/bin/python scripts/part2.py
```

Useful Colab/GPU command:

```bash
uv run python scripts/part2.py --skip-existing
```

Current default hyperparameters:

- Model: 5 hidden layer MLP, 256 hidden units, 128-dimensional sinusoidal time embedding.
- Optimizer: Adam.
- Learning rate: `1e-3`.
- Batch size: `1024`.
- Training steps: `25,000`.
- Sampling: Euler ODE from `t=1` to `t=0`.
- Sampling steps: `50`.
- Number of samples plotted: `4096`.
- Time clipping: `t in [1e-2, 1 - 1e-2]`.

Expected output pattern:

- Figures: `outputs/part2/generated_{dataset}_d{D}_{prediction}_pred_{loss}_loss.png`
- Checkpoints: `outputs/part2/model_{dataset}_d{D}_{prediction}_pred_{loss}_loss.pt`
- Results: `outputs/part2/part2_results.json`

### 2.3 Part 2 Questions

Answer:

- Which prediction type scales successfully to high ambient dimensions?
- At what dimension do the other prediction types visibly begin to fail?
- Does the choice of loss space (`x`-loss vs `v`-loss) affect which prediction types succeed or fail?
- What does this imply about what determines generation quality?
- Explain why the successful prediction type works at high `D` while the others do not.
- Discuss the rank/intrinsic-dimensionality idea:
  - Data manifold is intrinsically 2D.
  - `x` target lies on or near the low-dimensional data manifold.
  - `v = eps - x` contains full-dimensional noise information in ambient dimension `D`.

Paper/reference notes to mention:

- JiT separates prediction space from loss space: the network output can be `x`, `epsilon`, or `v`, while the loss can be measured in another space after conversion.
- The paper's main claim is that predicting clean data is qualitatively different from predicting noised quantities because clean data follows a low-dimensional manifold assumption, while noise and velocity are full-dimensional/off-manifold.
- The assignment's toy setup mirrors the JiT toy experiment: an intrinsic 2D dataset is embedded into larger ambient dimensions, and only `x`-prediction is expected to scale cleanly as `D` grows.

Fill after running full Part 2:

- Successful high-dimensional prediction type: `x`-prediction.
- First visible failure dimension for other prediction type: degradation begins to be visible at `D=8` for some `v`-prediction cases, and becomes clear failure at `D=32`.
- Does loss space change success/failure?: No. Loss space changes visual artifacts and sharpness, but the decisive factor is prediction type. Both `x`-pred/`x`-loss and `x`-pred/`v`-loss still preserve the target structures at `D=32`; both `v`-pred/`x`-loss and `v`-pred/`v`-loss fail or strongly degrade at `D=32`.
- Short explanation for report: The clean target `x` lies on the intrinsic 2D data manifold even when embedded in `D=32`, so the network learns a low-rank/low-dimensional target. The velocity target `v = eps - x` contains full-dimensional Gaussian noise components, so its complexity grows with ambient dimension. This makes direct `v`-prediction much harder at high `D`, regardless of whether the loss is measured in `x` or `v` space.

Observed Part 2 results:

- `D=2`: all four parameterization/loss combinations produce recognizable structures on all three datasets. There are small quality differences, but this dimension behaves as the warm-up sanity check predicts.
- `D=8`: `x`-prediction remains stable across datasets. `v`-prediction starts to show visible degradation:
  - `swiss_roll`: `v`-prediction still shows a spiral-like shape but is noisier and less clean than `x`-prediction.
  - `gaussians`: `v`-prediction finds modes but introduces ring-like bridges or extra mass between modes.
  - `circles`: `v`-prediction, especially `v`-pred/`x`-loss, fills in the space between rings and loses clean manifold structure.
- `D=32`: `x`-prediction succeeds and preserves the qualitative data structure for all datasets:
  - `swiss_roll`: spiral structure remains clear, with minor artifacts/outliers.
  - `gaussians`: eight modes remain distinct.
  - `circles`: two-ring structure remains clear.
- `D=32`: `v`-prediction fails visibly:
  - `swiss_roll`: generated samples become a noisy blob or partial circular mass instead of a clean spiral.
  - `gaussians`: generated samples form a large ring/connected structure rather than separated Gaussian modes.
  - `circles`: generated samples collapse into a noisy filled cloud, losing the two-ring manifold.
- Report files:
  - `outputs/part2/part2_results.json`
  - `outputs/part2/generated_{dataset}_d{D}_{prediction}_pred_{loss}_loss.png`

### 2.4 Answers to Part 2 Section 4.1 Questions

Question 1: Which prediction type scales successfully to high ambient dimensions? At what dimension do the other prediction types begin to fail visibly?

Answer draft:

- `x`-prediction scales successfully to high ambient dimensions in these experiments.
- At `D=32`, both `x`-pred/`x`-loss and `x`-pred/`v`-loss still produce recognizable samples for all three datasets.
- `v`-prediction begins to visibly degrade at `D=8`, especially on `circles` and `gaussians`, where generated samples start to fill in low-density regions or form bridges between modes.
- By `D=32`, `v`-prediction clearly fails:
  - `swiss_roll` becomes a noisy blob or partial ring rather than a clean spiral.
  - `gaussians` becomes a connected ring-like structure rather than eight separated modes.
  - `circles` becomes a filled noisy cloud rather than two clean rings.

Question 2: Does the choice of loss space (`x`-loss, `v`-loss) affect which prediction types succeed or fail? What does this tell you about what determines generation quality?

Answer draft:

- The loss space affects details such as sharpness, scatter, and the exact artifact pattern, but it does not change the main success/failure split.
- The decisive factor is prediction type:
  - Both `x`-prediction variants succeed at `D=32`.
  - Both `v`-prediction variants fail or strongly degrade at `D=32`.
- This suggests that generation quality is determined mainly by what the network is asked to predict directly, not only by the metric used to train it.
- In this assignment setup, predicting clean data `x` gives the model a structured low-dimensional target, while predicting velocity `v` asks the model to model full-dimensional noise-dependent quantities.

Question 3: Explain why the successful prediction type works at high `D` while the others do not. Consider the nature of each prediction target. Hint: think about the rank of each prediction target relative to the ambient dimension.

Answer draft:

- The data are intrinsically 2D even when embedded into `D=8` or `D=32`, because the 2D toy samples are projected into a higher-dimensional ambient space using an orthogonal projection.
- Therefore, the clean target `x` remains low-rank/low-dimensional relative to the ambient dimension.
- An `x`-prediction model only needs to learn the clean data manifold and map noisy points back toward that low-dimensional structure.
- In contrast, the velocity target `v = eps - x` includes the Gaussian noise `eps`, which has independent components across the full ambient dimension.
- As `D` increases, `v` becomes a full-dimensional target whose complexity grows with ambient dimension, even though the underlying data manifold remains 2D.
- This explains why `v`-prediction can work at `D=2` but fails as the ambient dimension increases: it is forced to learn high-rank noise-dominated targets, while `x`-prediction remains tied to the low-dimensional data structure.

## Part 3: Can We Rescue v-Prediction?

Main task:

- Based on Part 2 and Section 3 of the RAE paper, test whether `v`-prediction can be rescued at `D=32` on `swiss_roll`.
- Note: in the arXiv HTML version, the most relevant RAE discussion is in Section 4, especially:
  - width/capacity must scale with latent/token dimensionality;
  - high-dimensional latents benefit from dimension-dependent noise schedule shifts;
  - noisy generated latents create out-of-distribution issues, motivating smoother/noise-robust representations.

Report:

- Generated sample visualizations for each attempted rescue experiment.
- Clearly label which configurations succeeded and which failed.
- Explain what works, what does not, and why.

Questions to answer:

- Is `v`-prediction's failure fundamental, or can it be overcome?
- Do the Part 3 findings support or contradict the Part 2 observations?
- What approaches were tried?
- Compare compute cost between the rescue approach and the default `x`-prediction setup at `D=32`.
- Compare how `x`-prediction and `v`-prediction respond to the changes.
- Explain why they behave similarly or differently, considering what each parameterization must learn as ambient dimension `D` grows.
- Explain why real image generation systems such as Stable Diffusion 3 and FLUX can use `v`-prediction successfully even though it fails in these toy experiments.

Possible experiment notes to fill later:

- Approach tried:
- Training steps:
- Model size:
- Batch size:
- Learning rate:
- Result:
- Compute cost compared with default `x`-prediction:

Planned Part 3 experiments on `swiss_roll`, `D=32`:

- `ambient_v_default`
  - Purpose: reproduce the failed full-dimensional `v`-prediction baseline from Part 2.
  - Setup: full `D=32`, `v`-prediction, `v`-loss, 5 hidden layers, width 256, 25K steps, uniform time sampling.
  - Expected result: fail; noisy blob/partial ring rather than clean spiral.
- `ambient_v_long`
  - Purpose: test whether training compute alone rescues `v`-prediction.
  - Setup: same as default, but 100K steps.
  - Expected result: possible partial improvement, but likely still worse than `x`-prediction if target dimensionality is the main issue.
- `ambient_v_wide`
  - Purpose: test RAE-inspired capacity/width scaling.
  - Setup: full `D=32`, width 1024, 5 hidden layers, 50K steps.
  - Expected result: may improve full-dimensional `v`-prediction, but at significantly higher compute cost.
- `ambient_v_shift_high_noise`
  - Purpose: test RAE-inspired dimension-dependent noise schedule shift.
  - Setup: full `D=32`, width 256, 50K steps, time sampling shifted toward larger `t` values, i.e. more noisy inputs.
  - Expected result: may reduce some artifacts, but schedule alone may not solve full-rank target difficulty.
- `ambient_v_wide_shift_high_noise`
  - Purpose: combine width scaling and high-noise schedule shift.
  - Setup: full `D=32`, width 1024, 50K steps, shifted time sampling.
  - Expected result: strongest full-dimensional rescue attempt; compare quality and compute cost to default `x`-prediction.
- `projected_2d_v`
  - Purpose: test whether `v`-prediction works when the target is restricted to the intrinsic 2D manifold.
  - Setup: project `D=32` data back to 2D, train `v`-prediction in that 2D coordinate system, then embed generated samples back to `D=32` using the projection matrix.
  - Expected result: success; demonstrates that the failure is not because `v`-prediction is inherently impossible, but because ambient full-dimensional `v` contains high-rank noise components.

Part 3 output pattern:

- Figures: `outputs/part3/{experiment_name}.png`
- Checkpoints: `outputs/part3/{experiment_name}.pt`
- Results: `outputs/part3/part3_results.json`

Colab command:

```bash
uv run python scripts/part3.py --skip-existing --no-progress
```

## Part 4: One-Step Generation

### 4.1 Sampling Efficiency

Report:

- Use the best model from Part 2.
- Evaluate sample quality across Euler step counts:
  - `1`
  - `2`
  - `5`
  - `10`
  - `20`
  - `50`
  - `100`
  - `200`
- Include generated sample visualizations at each step count.
- Describe how quality degrades as step count decreases.

### 4.2 MeanFlow

Report:

- Implement MeanFlow using the best-performing prediction type from Part 2.
- Extend the model with horizon input `h = t - r`.
- Use a second 128-dimensional sinusoidal embedding for `h`.
- Use `torch.func.jvp` for the JVP computation.
- Use a flow matching ratio around `0.5` as a starting point:
  - 50% `h=0` standard flow matching.
  - 50% `h>0` mean velocity.
- Train at `D=32`.
- Compare:
  - MeanFlow samples with `1`, `2`, and `5` steps.
  - Multi-step standard flow matching samples.
- Use all three datasets.
- Include 9 MeanFlow figures:
  - `3 datasets * 3 step counts`.

### 4.3 Part 4 Questions

Answer:

- Why choose this prediction type for MeanFlow?
- Connect the choice to the Part 2 findings.
- In your own words, describe the core idea behind MeanFlow.
- What does MeanFlow learn that is different from standard flow matching?
- Why does this enable one-step generation?
- Why is the `h=0` portion needed during training?
- Compare MeanFlow training cost to standard flow matching.
- Why is MeanFlow harder to train?
- What is the computational overhead of the JVP operation per training step?
- Compare MeanFlow-generated samples against ground truth across all three datasets.
- Note any differences or artifacts, especially on the `gaussians` dataset.
- Explain why those artifacts occur, or why they do not.
