import { defineMcp } from "@lovable.dev/mcp-js";
import validateJsonTool from "./tools/validate-json";
import summarizeRunTool from "./tools/summarize-run";
import compareRunsTool from "./tools/compare-runs";

export default defineMcp({
  name: "multi-ml-classifier-mcp",
  title: "Multi-ML Classifier MCP",
  version: "0.1.0",
  instructions:
    "Stateless helpers for multi-ML-classifier results JSON files exported by the R pipeline. Use `validate_ml_results_json` to check a payload against the dashboard schema, `summarize_ml_run` to get run metadata + per-model AUROC/Accuracy/F1 + top selected features, and `compare_ml_runs` to diff two runs.",
  tools: [validateJsonTool, summarizeRunTool, compareRunsTool],
});