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

Observed Part 3 results:

- `ambient_v_default`
  - Result: failed.
  - Visual: still a noisy filled blob/partial ring, similar to the failed Part 2 `v`-prediction result at `D=32`.
  - Training cost: 25K steps, width 256, about 2.02 minutes on A100.
- `ambient_v_long`
  - Result: succeeded reasonably well.
  - Visual: clear swiss-roll spiral structure appears, although with more outliers/noise than default `x`-prediction.
  - Training cost: 100K steps, width 256, about 8.23 minutes on A100.
  - Interpretation: more optimization compute can partially overcome the full-dimensional `v` target.
- `ambient_v_wide`
  - Result: succeeded.
  - Visual: clear spiral with relatively few outliers; quality is close to the successful `x`-prediction results.
  - Training cost: 50K steps, width 1024, about 4.17 minutes on A100.
  - Interpretation: increasing model width/capacity is an effective rescue strategy, consistent with the RAE discussion that high-dimensional latents require wider networks.
- `ambient_v_shift_high_noise`
  - Result: failed.
  - Visual: noisy filled structure, no clean spiral.
  - Training cost: 50K steps, width 256, about 4.10 minutes on A100.
  - Interpretation: a timestep/noise schedule shift alone is not enough for this toy full-dimensional `v`-prediction problem.
- `ambient_v_wide_shift_high_noise`
  - Result: succeeded.
  - Visual: clear spiral, similar to `ambient_v_wide`, though still with some scatter/outliers.
  - Training cost: 50K steps, width 1024, about 4.14 minutes on A100.
  - Interpretation: schedule shift can work when paired with enough capacity, but width appears to be the key factor in this setup.
- `projected_2d_v`
  - Result: succeeded.
  - Visual: clean swiss-roll spiral after training `v`-prediction in the intrinsic 2D coordinate system and embedding samples back to `D=32`.
  - Training cost: 25K steps, width 256, about 2.56 minutes on A100.
  - Interpretation: `v`-prediction is not inherently impossible. The difficulty comes from predicting full-dimensional ambient velocity/noise components rather than the intrinsic 2D manifold dynamics.

### 3.1 Answers to Part 3 Section 5.1 Questions

Question 1: Based on your experiments, is `v`-prediction's failure fundamental or can it be overcome? Do your findings support or contradict your observations from Part 2?

Answer draft:

- The failure is not fundamental. It can be overcome, but not with the default model/training setup.
- Part 2 showed that default `v`-prediction fails at `D=32`, and Part 3 confirms this with `ambient_v_default`.
- However, Part 3 also shows that `v`-prediction can be rescued by:
  - substantially increasing training compute (`ambient_v_long`);
  - increasing model width/capacity (`ambient_v_wide`);
  - combining width with a shifted noise schedule (`ambient_v_wide_shift_high_noise`);
  - reducing the problem back to the intrinsic 2D subspace (`projected_2d_v`).
- This supports the Part 2 interpretation rather than contradicting it: `v`-prediction fails because the default setup is underpowered for the full-dimensional ambient target, not because `v`-prediction is mathematically impossible.

Question 2: What approach(es) did you try? Compare the compute cost between your approach and the default `x`-prediction setup to achieve similar sample quality at `D=32`.

Answer draft:

- Tried approaches:
  - longer training with default width (`ambient_v_long`);
  - wider MLP (`ambient_v_wide`);
  - high-noise timestep shift (`ambient_v_shift_high_noise`);
  - wider MLP plus high-noise shift (`ambient_v_wide_shift_high_noise`);
  - intrinsic 2D projection before training `v`-prediction (`projected_2d_v`).
- Default `x`-prediction at `D=32` from Part 2 used 25K steps and width 256, and already produced good samples.
- To get comparable full-dimensional `v`-prediction quality:
  - `ambient_v_long` needed 100K steps, about 4x the number of training steps.
  - `ambient_v_wide` needed width 1024 and 50K steps, meaning a much larger model and about 2x the training steps.
  - `ambient_v_wide_shift_high_noise` had similar cost to `ambient_v_wide`.
- `projected_2d_v` achieved good quality with 25K steps and width 256, but it changes the problem by using knowledge of the true intrinsic 2D subspace. It is useful as a diagnostic experiment, but less realistic as a general high-dimensional generative modeling method.

Question 3: Compare how `x`-prediction and `v`-prediction respond to your changes. Do they behave the same way? Explain why or why not.

Answer draft:

- They do not behave the same way.
- `x`-prediction already works well at `D=32` with the default 25K-step, width-256 model.
- `v`-prediction needs additional help: longer training, wider networks, or projection into the intrinsic subspace.
- The difference comes from the target each model must learn:
  - `x`-prediction learns a low-dimensional clean data target, because the swiss roll is intrinsically 2D even when embedded in `D=32`.
  - `v`-prediction learns `v = eps - x`, which includes full-dimensional Gaussian noise components in the ambient 32D space.
- Increasing width or training time helps `v`-prediction because it gives the model more capacity/optimization budget for this higher-rank target. `x`-prediction does not need as much extra capacity because its target remains tied to the low-dimensional manifold.

Question 4: In practice, `v`-prediction is used successfully in real image generation systems such as Stable Diffusion 3 and FLUX. Why might the situation be different for those models compared to these toy datasets?

Answer draft:

- Real image systems differ from this toy setup in several important ways.
- Their latent spaces are learned by autoencoders or representation encoders, not created by embedding a 2D manifold into a mostly empty high-dimensional ambient space.
- The latent dimensions in image models contain meaningful semantic/spatial structure rather than arbitrary orthogonal null directions.
- Modern diffusion/flow models such as SD3 and FLUX use much larger architectures and much larger training budgets than the small MLP used here.
- Their noise schedules, normalization, latent scaling, and training objectives are heavily tuned for the latent representation.
- RAE also argues that high-dimensional representation spaces can work when the architecture is adjusted, especially by increasing width/capacity and handling noisy latents properly.
- Therefore, the toy failure of default `v`-prediction reflects a mismatch between a small model/default training setup and a full-dimensional ambient target, not a universal rule that `v`-prediction cannot work in real generative systems.

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

Implementation/output notes:

- Implemented in `scripts/part4_sampling_efficiency.py`.
- Uses the Part 2 best-performing parameterization: `x`-prediction at `D=32`.
- Current checkpoint choice: `x`-pred/`v`-loss for each dataset, since this gave the cleanest qualitative Part 2 samples at `D=32` overall.
- Evaluated datasets:
  - `swiss_roll`
  - `gaussians`
  - `circles`
- Step counts:
  - `1`, `2`, `5`, `10`, `20`, `50`, `100`, `200`
- Output directory:
  - `outputs/part4/sampling_efficiency_x_pred_v_loss/`
- Key files:
  - `outputs/part4/sampling_efficiency_x_pred_v_loss/swiss_roll_d32_steps_grid.png`
  - `outputs/part4/sampling_efficiency_x_pred_v_loss/gaussians_d32_steps_grid.png`
  - `outputs/part4/sampling_efficiency_x_pred_v_loss/circles_d32_steps_grid.png`
  - `outputs/part4/sampling_efficiency_x_pred_v_loss/sampling_efficiency_results.json`

Observed 6.1 results:

- `1` Euler step:
  - Fails for all datasets.
  - Samples remain a collapsed/noisy central cloud and do not match the target distribution.
- `2` Euler steps:
  - Still fails.
  - Samples spread out more than 1 step, but remain diffuse and off-manifold.
- `5` Euler steps:
  - Coarse global structure begins to appear.
  - `swiss_roll`: partial circular/spiral-like shape appears, but with heavy noise.
  - `gaussians`: modes start to arrange around the correct ring, but are connected by many off-manifold points.
  - `circles`: two-ring structure begins to emerge, but the space between rings is heavily filled.
- `10` Euler steps:
  - Structure is recognizable for all three datasets.
  - Still visibly noisy and less sharp than 50+ step sampling.
- `20` Euler steps:
  - Quality becomes much better.
  - Most target structures are recognizable with moderate remaining scatter.
- `50` Euler steps:
  - This is the Part 2 default and gives good quality.
  - Samples are close to the expected structures.
- `100` and `200` Euler steps:
  - Similar or slightly cleaner than 50 steps.
  - Improvements over 50 steps are modest compared with the large jump from 5 to 20 steps.

Answer draft for 6.1:

- Standard flow matching with Euler sampling is not a reliable one-step generator in this setup.
- Very few Euler steps produce poor samples because each model evaluation predicts only an instantaneous velocity field; a single large step is a crude approximation to the full ODE trajectory.
- Quality improves rapidly as the number of integration steps increases.
- Around 20 steps, the main structures are already recognizable.
- Around 50 steps, sample quality is close to the Part 2 baseline, and 100/200 steps provide diminishing returns.

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

Implementation/output notes:

- Implemented model extension in `src/model.py`:
  - `MeanFlowMLP(data_dim, time_embed_dim=128, hidden_dim=256, hidden_layers=5)`.
  - Inputs are concatenated as `[z_t, time_embedding(t), horizon_embedding(h)]`.
  - First hidden layer input size is `D + 256` when both embeddings are 128-dimensional.
- Implemented training and sampling utilities in `src/flow_matching.py`:
  - `train_mean_flow(...)`
  - `sample_mean_flow(...)`
- Implemented experiment runner in `scripts/part4_meanflow.py`.
- MeanFlow training target follows the paper's average-velocity identity:
  - Assignment convention: clean data at `t=0`, noise at `t=1`.
  - Forward path: `z_t = x + t v`, where `v = eps - x`.
  - Model output: interval average velocity `u_theta(z_t, t, h)`.
  - Use `torch.func.jvp` to compute the total derivative of `u_theta` along increasing `t` at fixed interval start `r`, with tangent `(dz/dt, dt/dt, dh/dt) = (v, 1, 1)`.
  - Training target: `stopgrad(v - h * d u_theta / dt)`.
  - Loss: `MSE(u_theta, stopgrad(v - h * d u_theta / dt))`.
- MeanFlow sampling:
  - Start from Gaussian noise at `t=1`.
  - For one step, use `h=1` and update `z_0 = z_1 - u_theta(z_1, 1, 1)`.
  - For `K` steps, use step size `h=1/K` and repeatedly update `z <- z - h u_theta(z, t, h)`.
- Prediction-type note:
  - Part 2's successful high-dimensional parameterization was `x`-prediction, with `x`-pred/`v`-loss giving the best qualitative D=32 baseline used in 6.1.
  - MeanFlow itself learns an average velocity rather than directly outputting clean `x`.
  - In the report, connect the choice to Part 2 as follows: we use the Part 2 `x`-prediction result to choose the D=32 baseline and to motivate why the clean-data/manifold structure matters, but the MeanFlow objective replaces instantaneous Euler velocity prediction with interval average velocity prediction for one-step sampling.

Default Part 4.2 command:

```bash
uv run python scripts/part4_meanflow.py --no-progress
```

Useful Colab/GPU command:

```bash
uv run python scripts/part4_meanflow.py --skip-existing --no-progress
```

Default MeanFlow hyperparameters:

- Datasets: `swiss_roll`, `gaussians`, `circles`.
- Dimension: `D=32`.
- Model: 5 hidden layer MLP, 256 hidden units, 128-dimensional sinusoidal embedding for `t`, 128-dimensional sinusoidal embedding for `h`.
- Optimizer: Adam.
- Learning rate: `1e-3`.
- Batch size: `1024`.
- Training steps: `50,000`.
- Time clipping: `t in [1e-2, 1 - 1e-2]`.
- Flow matching ratio: `0.5`.
  - 50% examples use `h=0`.
  - 50% examples use `h>0`.
- Number of samples plotted: `4096`.
- MeanFlow sampling step counts: `1`, `2`, `5`.
- Standard flow matching comparison step counts: `1`, `2`, `5`, `10`, `20`, `50`.
- Standard flow matching baseline: Part 2 `x`-pred/`v`-loss checkpoints.

Expected output pattern:

- MeanFlow figures:
  - `outputs/part4/meanflow/meanflow_{dataset}_d32_1_steps.png`
  - `outputs/part4/meanflow/meanflow_{dataset}_d32_2_steps.png`
  - `outputs/part4/meanflow/meanflow_{dataset}_d32_5_steps.png`
- MeanFlow per-dataset grids:
  - `outputs/part4/meanflow/meanflow_{dataset}_d32_steps_grid.png`
- MeanFlow vs standard flow matching comparison grids:
  - `outputs/part4/meanflow/comparison_{dataset}_d32.png`
- Checkpoints:
  - `outputs/part4/meanflow/meanflow_{dataset}_d32.pt`
- Results:
  - `outputs/part4/meanflow/meanflow_results.json`

Planned Part 4.2 comparisons:

- For each dataset, compare:
  - MeanFlow at `1`, `2`, and `5` sampling steps.
  - Standard flow matching at `1`, `2`, `5`, `10`, `20`, and `50` Euler steps.
- Main success criterion:
  - MeanFlow should produce recognizable samples at one or very few steps.
  - Standard flow matching from 6.1 needs many more Euler steps, with 1-2 steps failing and 20-50 steps becoming good.
- Dataset-specific observations to check:
  - `swiss_roll`: whether one-step MeanFlow preserves the spiral rather than making a circular blob.
  - `gaussians`: whether one-step MeanFlow keeps separated modes or produces bridges between modes.
  - `circles`: whether one-step MeanFlow preserves two rings without filling the middle.

Observed 6.2 results:

- Fill after running `scripts/part4_meanflow.py` on Colab/A100.
- Record per dataset:
  - One-step quality:
  - Two-step quality:
  - Five-step quality:
  - Comparison against standard flow matching:
  - Artifacts:
  - Approximate training time:

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

### 4.3 Answers to Part 4 Section 6.3 Questions

Question 1: Why did you choose this prediction type for MeanFlow? Connect your choice to Part 2 findings.

Answer draft:

- Part 2 showed that prediction target matters more than loss space at high ambient dimension.
- At `D=32`, `x`-prediction succeeded while direct `v`-prediction failed under the default training setup.
- For the 6.1 baseline, the best qualitative D=32 checkpoints were `x`-pred/`v`-loss, so the standard flow matching comparison uses those checkpoints.
- MeanFlow changes the sampling objective: it learns an interval average velocity rather than relying on an instantaneous velocity field plus many Euler steps.
- The reason this is sensible after Part 2 is that the clean-data manifold structure remains important at high `D`; one-step generation should avoid asking the sampler to recover the whole trajectory from a single crude instantaneous Euler step.
- Important nuance for the report: MeanFlow's network output is an average velocity `u`, not a direct clean sample `x`. The Part 2 result motivates the baseline/modeling choice and interpretation, while MeanFlow's own objective is the average-velocity objective from the paper.

Question 2: In your own words, describe the core idea behind MeanFlow. What does it learn that is different from standard flow matching? Why does this enable one-step generation?

Answer draft:

- Standard flow matching learns an instantaneous velocity field at each time.
- Sampling then approximates the ODE trajectory by taking many small Euler steps.
- MeanFlow instead learns the average velocity over a time interval, parameterized by the current time `t` and horizon `h=t-r`.
- If the model can predict the average velocity from noise time `t=1` all the way to clean time `t=0`, then a single update can jump directly from noise to data.
- Multi-step MeanFlow is still possible by using smaller horizons, but the objective explicitly trains the model to handle nonzero intervals, including the full one-step interval.

Question 3: Why is the `h=0` portion needed during training?

Answer draft:

- When `h=0`, the interval average velocity reduces to the ordinary instantaneous velocity.
- Including `h=0` anchors MeanFlow to the standard flow matching target and stabilizes training.
- It gives the model many local, easier supervision examples while the `h>0` examples teach it to predict longer-interval average velocities.
- Without the `h=0` portion, the model would rely entirely on the harder JVP-derived mean-velocity targets, which can be noisier and less stable early in training.

Question 4: Compare MeanFlow training cost to standard flow matching. Why is MeanFlow harder to train? What is the computational overhead of the JVP operation per training step?

Answer draft:

- Standard flow matching needs one model evaluation per training batch.
- MeanFlow needs the model output and a Jacobian-vector product of the output with respect to `(z,t,h)`.
- In PyTorch, `torch.func.jvp` has roughly the cost of an additional forward-mode derivative computation, so each MeanFlow step is noticeably more expensive than a standard flow matching step.
- MeanFlow is also harder because the target depends on the model's own derivative `d u_theta / dt`, then uses a stop-gradient target.
- This creates a more complex training signal than directly regressing to `x` or `v`.
- In the final report, fill in measured A100 wall-clock times from `outputs/part4/meanflow/meanflow_results.json` and compare them to the Part 2/Part 4.1 training/sampling costs.

Question 5: Compare MeanFlow-generated samples against ground truth across all three datasets. Note differences or artifacts, especially on `gaussians`.

Answer draft:

- Fill after running 6.2.
- Expected discussion points:
  - Whether one-step samples already show the correct global structure.
  - Whether 2-step or 5-step sampling improves sharpness and removes off-manifold scatter.
  - Whether `gaussians` has mode-bridging artifacts, mode collapse, or blurred modes.
  - If `gaussians` artifacts appear, a likely reason is that one interval average velocity can smooth across separated modes, especially where the trajectory must choose between multiple disconnected clusters.
  - If artifacts do not appear, state that the learned average velocity separated the modes well, and support it with the saved figures.
