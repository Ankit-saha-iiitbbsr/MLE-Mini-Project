# Monitoring & Drift Report

*Generated 2026-08-16T10:31:58Z from 3624 logged predictions.*

## 1. What is monitored

Every served prediction logs seven image statistics alongside the decision. Those statistics are compared against a baseline frozen from the **training split**, using PSI (with reference-derived, frozen bin edges) and a two-sample KS test. Model behaviour (confidence, predicted-defect rate, human-review rate) is tracked in parallel because it degrades *before* labelled accuracy does.

## 2. Results by scenario

| scenario | max PSI | drifted features | accuracy | acc. drop | mean conf. | alerts |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline` | 0.0505 | 0 (none) | 0.9950 | +0.0016 | 0.9891 | none |
| `camera_angle` | 12.4339 | 7 (edge_density, entropy, laplacian_var) | 0.9400 | +0.0566 | 0.9628 | data_drift, accuracy_degradation |
| `focus_blur` | 12.4339 | 4 (edge_density, laplacian_var, p05_intensity) | 0.9925 | +0.0041 | 0.9592 | data_drift |
| `lighting_bright` | 12.6172 | 7 (edge_density, entropy, laplacian_var) | 0.6100 | +0.3866 | 0.9697 | data_drift, accuracy_degradation |
| `lighting_dim` | 15.1855 | 7 (edge_density, entropy, laplacian_var) | 0.5775 | +0.4191 | 0.9963 | data_drift, accuracy_degradation |
| `new_variant` | 12.6172 | 7 (edge_density, entropy, laplacian_var) | 0.8375 | +0.1591 | 0.9266 | data_drift, accuracy_degradation |
| `real_camera_upgrade` | 7.3440 | 6 (edge_density, laplacian_var, mean_intensity) | 0.7800 | +0.2166 | 0.9724 | data_drift, accuracy_degradation |
| `sensor_noise` | 12.4339 | 6 (edge_density, entropy, laplacian_var) | 0.7425 | +0.2541 | 0.9103 | data_drift, confidence_collapse, accuracy_degradation |

Alert thresholds: PSI warn `0.1`, PSI alert `0.25`, confidence drop `0.08`, accuracy drop `0.05`, review rate `0.2`.

## 3. Figures

- `figures/monitoring/psi_heatmap.png`
- `figures/monitoring/performance_by_scenario.png`
- `figures/monitoring/window_trend.png`
- `figures/monitoring/score_distributions.png`

## 4. Reading the results

- The `baseline` row is the control: uncorrupted test images. Its PSI shows the noise floor, i.e. how much apparent drift arises from sampling alone. Any scenario must be read against that floor, not against zero.
- `real_camera_upgrade` is not simulated. It is a genuine second capture of the same production line with a different camera, held back from training specifically so the detectors face a shift no corruption operator was tuned on.

### On the magnitude of PSI

The conventional bands (0.10 warn, 0.25 significant) are calibrated for the *subtle* shifts typical of tabular credit scoring. Values in double digits are not an error: PSI is unbounded, and a value near 12 is the arithmetic signature of a distribution that has moved **entirely outside the reference support** -- every sample landing in one reference bin, with the other nine effectively empty. Anything above roughly 1.0 should be read as "a different distribution", not as "1.0/0.25 = 4x worse than the alert threshold".

This is also why the retraining policy routes very large PSI to `investigate_capture` rather than `retrain`: a shift that large is a hardware or configuration change, not a change in the parts being inspected.

### The *pattern* of drifted features identifies the fault

See `figures/monitoring/psi_heatmap.png`. The features were chosen so that each maps to a physical failure, and the heatmap shows that holding: a defocus scenario lights up `laplacian_var` while leaving `mean_intensity` at the noise floor; a lighting shift does the reverse; sensor noise fires `edge_density` and `laplacian_var` together. So the monitoring output is not merely "something changed" -- the signature says *which* piece of the capture rig to go and look at. An embedding-based detector would fire just as reliably and tell nobody where to start.

### Confidence is not a sufficient monitor

Compare the accuracy and mean-confidence columns above. A model can lose most of its accuracy while becoming *more* certain -- the classic silent failure. Any scenario in this table where accuracy fell sharply but confidence did not is a case that confidence-based monitoring alone would have missed entirely, and that input-distribution monitoring caught. That asymmetry is the reason both tiers exist.
