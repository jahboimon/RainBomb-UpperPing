(() => {
  "use strict";

  function scalarBaseScore(raw) {
    const parsed = Number(String(raw).replace(/[\[\]]/g, ""));
    if (!Number.isFinite(parsed) || parsed <= 0 || parsed >= 1) throw new Error("Unsupported XGBoost base_score");
    return parsed;
  }

  function create(model) {
    const learner = model?.learner;
    if (!learner || learner.objective?.name !== "binary:logistic") throw new Error("Only the verified binary:logistic model is supported");
    const featureNames = learner.feature_names;
    const trees = learner.gradient_booster?.model?.trees;
    if (!Array.isArray(featureNames) || featureNames.length !== 20 || !Array.isArray(trees)) throw new Error("Unexpected frozen model structure");
    const baseScore = scalarBaseScore(learner.learner_model_param.base_score);
    const baseMargin = Math.log(baseScore / (1 - baseScore));

    function predict(features) {
      let margin = baseMargin;
      for (const tree of trees) {
        let node = 0;
        while (tree.left_children[node] !== -1) {
          const feature = featureNames[tree.split_indices[node]];
          const rawValue = Number(features[feature]);
          const value = Math.fround(rawValue);
          const missing = !Number.isFinite(value);
          const threshold = Math.fround(tree.split_conditions[node]);
          const goLeft = missing ? Boolean(tree.default_left[node]) : value < threshold;
          node = goLeft ? tree.left_children[node] : tree.right_children[node];
        }
        margin += tree.split_conditions[node];
      }
      return 1 / (1 + Math.exp(-margin));
    }

    return { featureNames: [...featureNames], treeCount: trees.length, predict };
  }

  function validate(runtime, payload) {
    let maxAbsoluteError = 0;
    for (const row of payload.rows || []) {
      const error = Math.abs(runtime.predict(row.features) - Number(row.saved_score));
      if (error > maxAbsoluteError) maxAbsoluteError = error;
    }
    const tolerance = Number(payload.tolerance);
    return {
      rowCount: (payload.rows || []).length,
      maxAbsoluteError,
      tolerance,
      status: maxAbsoluteError <= tolerance ? "PASS" : "FAIL"
    };
  }

  window.RainBombXGB = { create, validate };
})();
