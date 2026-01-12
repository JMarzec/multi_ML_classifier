import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Beaker, AlertCircle, CheckCircle2, XCircle, RefreshCw } from "lucide-react";
import type { MLResults, ModelPerformance } from "@/types/ml-results";

interface SamplePredictionTabProps {
  data: MLResults;
}

interface FeatureInput {
  name: string;
  value: string;
}

interface PredictionResult {
  model: string;
  probability: number;
  prediction: "Positive" | "Negative";
  confidence: number;
}

export function SamplePredictionTab({ data }: SamplePredictionTabProps) {
  const features = useMemo(() => {
    return data.selected_features?.length > 0 
      ? data.selected_features 
      : data.feature_importance?.slice(0, 10).map(f => f.feature) || [];
  }, [data.selected_features, data.feature_importance]);

  const [featureInputs, setFeatureInputs] = useState<FeatureInput[]>(
    features.map(name => ({ name, value: "" }))
  );
  const [predictions, setPredictions] = useState<PredictionResult[] | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const modelLabels: Record<string, string> = {
    rf: "Random Forest",
    svm: "SVM",
    xgboost: "XGBoost",
    knn: "KNN",
    mlp: "MLP",
    hard_vote: "Hard Voting",
    soft_vote: "Soft Voting",
  };

  const handleInputChange = (index: number, value: string) => {
    const newInputs = [...featureInputs];
    newInputs[index].value = value;
    setFeatureInputs(newInputs);
  };

  const handlePredict = () => {
    setIsProcessing(true);
    
    // Simulate prediction based on model performance data
    // In a real scenario, this would call a backend API with the trained models
    setTimeout(() => {
      const availableModels = Object.entries(data.model_performance)
        .filter(([, metrics]) => metrics?.auroc)
        .map(([model]) => model);

      const simulatedPredictions: PredictionResult[] = availableModels.map(model => {
        const metrics = data.model_performance[model as keyof ModelPerformance];
        const baseProb = metrics?.auroc?.mean || 0.5;
        
        // Add some variation based on input values
        const inputSum = featureInputs.reduce((sum, input) => {
          const val = parseFloat(input.value) || 0;
          return sum + val;
        }, 0);
        
        // Create realistic-looking probability variation
        const variation = (Math.sin(inputSum * 0.1) * 0.1) + (Math.random() * 0.05 - 0.025);
        const probability = Math.max(0, Math.min(1, baseProb + variation));
        
        return {
          model,
          probability,
          prediction: probability > 0.5 ? "Positive" as const : "Negative" as const,
          confidence: Math.abs(probability - 0.5) * 2,
        };
      });

      setPredictions(simulatedPredictions);
      setIsProcessing(false);
    }, 1000);
  };

  const handleReset = () => {
    setFeatureInputs(features.map(name => ({ name, value: "" })));
    setPredictions(null);
  };

  const fillRandomValues = () => {
    const newInputs = featureInputs.map(input => ({
      ...input,
      value: (Math.random() * 10 - 5).toFixed(2),
    }));
    setFeatureInputs(newInputs);
  };

  const allInputsFilled = featureInputs.every(input => input.value !== "");
  
  const ensemblePrediction = predictions 
    ? predictions.reduce((sum, p) => sum + p.probability, 0) / predictions.length
    : null;

  return (
    <div className="space-y-6">
      {/* Info Banner */}
      <div className="bg-info/10 border border-info/20 rounded-xl p-4 flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-info mt-0.5" />
        <div>
          <h4 className="font-medium text-foreground">Sample Prediction (Simulation Mode)</h4>
          <p className="text-sm text-muted-foreground mt-1">
            Enter feature values below to get predicted class probabilities from all trained models.
            Note: This is a simulated prediction based on model performance metrics. 
            For real predictions, use the trained models in R.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Feature Input Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Beaker className="w-5 h-5 text-primary" />
                Feature Values
              </span>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={fillRandomValues}>
                  Random
                </Button>
                <Button variant="outline" size="sm" onClick={handleReset}>
                  <RefreshCw className="w-4 h-4" />
                </Button>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-h-[400px] overflow-y-auto pr-2">
              {featureInputs.map((input, index) => (
                <div key={input.name} className="space-y-1">
                  <Label htmlFor={input.name} className="text-xs text-muted-foreground truncate block">
                    {input.name}
                  </Label>
                  <Input
                    id={input.name}
                    type="number"
                    step="0.01"
                    placeholder="0.00"
                    value={input.value}
                    onChange={(e) => handleInputChange(index, e.target.value)}
                    className="h-8 text-sm"
                  />
                </div>
              ))}
            </div>

            <div className="mt-6 flex gap-2">
              <Button 
                onClick={handlePredict} 
                disabled={!allInputsFilled || isProcessing}
                className="flex-1"
              >
                {isProcessing ? "Processing..." : "Get Predictions"}
              </Button>
            </div>
            
            {!allInputsFilled && (
              <p className="text-xs text-muted-foreground mt-2 text-center">
                Fill all {featureInputs.length} feature values to get predictions
              </p>
            )}
          </CardContent>
        </Card>

        {/* Predictions Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Model Predictions
              {ensemblePrediction !== null && (
                <Badge variant={ensemblePrediction > 0.5 ? "default" : "secondary"}>
                  Ensemble: {(ensemblePrediction * 100).toFixed(1)}%
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {predictions ? (
              <div className="space-y-4">
                {predictions.map((pred) => (
                  <div 
                    key={pred.model} 
                    className="bg-muted/30 rounded-lg p-4 border border-border"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">
                        {modelLabels[pred.model] || pred.model}
                      </span>
                      <div className="flex items-center gap-2">
                        {pred.prediction === "Positive" ? (
                          <CheckCircle2 className="w-4 h-4 text-green-500" />
                        ) : (
                          <XCircle className="w-4 h-4 text-red-500" />
                        )}
                        <Badge 
                          variant={pred.prediction === "Positive" ? "default" : "secondary"}
                          className="text-xs"
                        >
                          {pred.prediction}
                        </Badge>
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Probability</span>
                        <span className="font-mono">
                          {(pred.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <Progress 
                        value={pred.probability * 100} 
                        className="h-2"
                      />
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Confidence</span>
                        <span className="font-mono text-primary">
                          {(pred.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}

                {/* Ensemble Summary */}
                {ensemblePrediction !== null && (
                  <div className="bg-gradient-to-r from-primary/10 via-secondary/10 to-accent/10 rounded-lg p-4 border border-primary/20">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">Ensemble Prediction</p>
                        <p className="text-2xl font-bold">
                          {ensemblePrediction > 0.5 ? "Positive" : "Negative"}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-sm text-muted-foreground">Average Probability</p>
                        <p className="text-3xl font-bold text-primary">
                          {(ensemblePrediction * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-[300px] flex flex-col items-center justify-center text-center">
                <Beaker className="w-12 h-12 text-muted-foreground mb-4" />
                <p className="text-muted-foreground">
                  Enter feature values and click "Get Predictions" to see model outputs
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}