import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, CheckCircle, Info } from "lucide-react";

export function MLMethodInfoPanel() {
  const methods = [
    {
      name: "Random Forest (RF)",
      icon: "🌲",
      strengths: ["Robust on small datasets", "Handles high-dimensional data", "Built-in feature importance"],
      weaknesses: ["Can overfit with very noisy features"],
      dataSize: "small",
      note: "Recommended for most biological/clinical datasets.",
    },
    {
      name: "Support Vector Machine (SVM)",
      icon: "📐",
      strengths: ["Effective in high dimensions", "Memory efficient", "Works well with clear margins"],
      weaknesses: ["Sensitive to feature scaling", "Slow on large datasets"],
      dataSize: "small",
      note: "Good for small-to-medium datasets with clear class separation.",
    },
    {
      name: "K-Nearest Neighbors (KNN)",
      icon: "👥",
      strengths: ["Simple and interpretable", "No training phase", "Non-parametric"],
      weaknesses: ["Sensitive to irrelevant features", "Slow prediction on large datasets"],
      dataSize: "small",
      note: "Best with well-separated classes and few features.",
    },
    {
      name: "XGBoost",
      icon: "🚀",
      strengths: ["State-of-the-art performance", "Handles missing values", "Built-in regularization"],
      weaknesses: ["Requires large datasets for best results", "Many hyperparameters"],
      dataSize: "large",
      note: "Optimal for large datasets (>1000 samples). May underperform on small data.",
    },
    {
      name: "Multi-Layer Perceptron (MLP)",
      icon: "🧠",
      strengths: ["Captures complex non-linear patterns", "Flexible architecture"],
      weaknesses: ["Requires large datasets", "Prone to overfitting on small data", "Black-box model"],
      dataSize: "large",
      note: "Best for large datasets. Consider other methods for <500 samples.",
    },
  ];

  const ensembleMethods = [
    {
      name: "Hard Voting",
      description: "Each model votes for a class; majority wins.",
      warning: "Fails if any base learner collapses (predicts single class).",
      recommendation: "Use soft voting for robustness.",
    },
    {
      name: "Soft Voting",
      description: "Averages probability predictions across models.",
      warning: "More robust but requires probability outputs.",
      recommendation: "Recommended for small datasets — handles weak learners gracefully.",
    },
  ];

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Info className="w-5 h-5" />
            ML Method Selection Guide
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Choose appropriate methods based on your dataset size and characteristics.
          </p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {methods.map((method) => (
              <div
                key={method.name}
                className="bg-muted/30 rounded-xl p-4 border border-border"
              >
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-2xl">{method.icon}</span>
                  <div className="flex-1">
                    <h4 className="font-semibold text-sm">{method.name}</h4>
                    <Badge
                      variant={method.dataSize === "small" ? "default" : "secondary"}
                      className="text-xs mt-1"
                    >
                      {method.dataSize === "small" ? "Small Data OK" : "Needs Large Data"}
                    </Badge>
                  </div>
                </div>

                <div className="space-y-2 text-xs">
                  <div>
                    <p className="text-muted-foreground font-medium mb-1">Strengths:</p>
                    <ul className="space-y-0.5">
                      {method.strengths.map((s, i) => (
                        <li key={i} className="flex items-start gap-1">
                          <CheckCircle className="w-3 h-3 text-success mt-0.5 shrink-0" />
                          <span>{s}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <p className="text-muted-foreground font-medium mb-1">Weaknesses:</p>
                    <ul className="space-y-0.5">
                      {method.weaknesses.map((w, i) => (
                        <li key={i} className="flex items-start gap-1">
                          <AlertTriangle className="w-3 h-3 text-warning mt-0.5 shrink-0" />
                          <span>{w}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                <p className="text-xs text-info mt-3 italic">{method.note}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Ensemble Methods */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Ensemble Voting Strategies</CardTitle>
          <p className="text-sm text-muted-foreground">
            How individual model predictions are combined for final classification.
          </p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {ensembleMethods.map((ens) => (
              <div key={ens.name} className="bg-muted/30 rounded-xl p-4 border border-border">
                <h4 className="font-semibold mb-2">{ens.name}</h4>
                <p className="text-sm text-muted-foreground mb-2">{ens.description}</p>
                <div className="flex items-start gap-2 text-xs text-warning mb-2">
                  <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>{ens.warning}</span>
                </div>
                <div className="flex items-start gap-2 text-xs text-success">
                  <CheckCircle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>{ens.recommendation}</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Data Size Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Dataset Size Recommendations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-auto rounded-lg border border-border">
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr>
                  <th className="text-left p-3">Dataset Size</th>
                  <th className="text-left p-3">Recommended Methods</th>
                  <th className="text-left p-3">Avoid</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t border-border">
                  <td className="p-3 font-medium">Small ({"<"}100 samples)</td>
                  <td className="p-3">RF, SVM, KNN</td>
                  <td className="p-3 text-muted-foreground">XGBoost, MLP</td>
                </tr>
                <tr className="border-t border-border">
                  <td className="p-3 font-medium">Medium (100-500 samples)</td>
                  <td className="p-3">RF, SVM, XGBoost (with regularization)</td>
                  <td className="p-3 text-muted-foreground">MLP (may overfit)</td>
                </tr>
                <tr className="border-t border-border">
                  <td className="p-3 font-medium">Large ({">"}500 samples)</td>
                  <td className="p-3">XGBoost, MLP, RF, SVM</td>
                  <td className="p-3 text-muted-foreground">—</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-xs text-muted-foreground mt-3">
            <strong>Tip:</strong> This dashboard uses soft voting by default to ensure ensemble predictions
            remain stable even if one base learner fails due to small sample size.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
