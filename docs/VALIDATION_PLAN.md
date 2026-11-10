# Validation Plan

1. **Venue Perspective Calibration**: Measure the visible ground floor area (`monitored_area_m2`) and calibrate top/mid/bottom row weights per camera view so `people/m^2` and Fruin LOS levels match real physical density.
2. **Group-Separated Evaluation**: Always separate training/validation/test data by `video_id` and physical venue scene (`GroupShuffleSplit`); never split adjacent temporal windows across splits.
3. **Component Benchmarks**:
   - Evaluate person counting MAE/RMSE and multi-object tracking IDF1/MOTA against annotated venue sequences.
   - Evaluate panic dispersion, counter-flow, and crush stagnation triggers at the **event level** (precision, recall, and false alarms per camera-hour).
   - Evaluate +30s forecast lead time, PR-AUC, and calibration against ground-truth onset timestamps when available.
4. **Provisional Thresholds**: Treat default physics thresholds (`panic_score >= 55`, `predicted_score_30s >= 70`, `Fruin LOS E/F`) as baseline heuristics to tune via shadow-mode site trials before operational reliance.
