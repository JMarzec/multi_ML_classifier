import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from "recharts";
import type { MLResults } from "@/types/ml-results";

interface CalibrationCurveTabProps {
  data: MLResults;
}

interface CalibrationPoint {
  binCenter: number;
  rf: number | null;
  svm: number | null;
  xgboost: number | null;
  knn: number | null;
  mlp: number | null;
  ensemble: number | null;
  perfect: number;
}

export function CalibrationCurveTab({ data }: CalibrationCurveTabProps) {
  // Generate simulated calibration data based on model performance
  // In real implementation, this would come from the R script
  const generateCalibrationData = (): CalibrationPoint[] => {
    const bins = 10;
    const points: CalibrationPoint[] = [];
    
    // Get model accuracies to simulate calibration quality
    const modelAccuracies = {
      rf: data.model_performance.rf?.accuracy?.mean || 0.8,
      svm: data.model_performance.svm?.accuracy?.mean || 0.75,
      xgboost: data.model_performance.xgboost?.accuracy?.mean || 0.82,
      knn: data.model_performance.knn?.accuracy?.mean || 0.72,
      mlp: data.model_performance.mlp?.accuracy?.mean || 0.78,
      ensemble: data.model_performance.soft_vote?.accuracy?.mean || 0.85,
    };
    
    for (let i = 0; i < bins; i++) {
      const binCenter = (i + 0.5) / bins;
      
      // Simulate calibration based on model accuracy
      // Better models are more calibrated (closer to perfect diagonal)
      const addNoise = (acc: number, center: number) => {
        const calibrationError = (1 - acc) * 0.3;
        const noise = (Math.random() - 0.5) * calibrationError;
        // Models tend to be overconfident at high probabilities
        const overconfidence = center > 0.5 ? (center - 0.5) * 0.1 : 0;
        return Math.max(0, Math.min(1, center + noise - overconfidence));
      };
      
      points.push({
        binCenter: parseFloat((binCenter * 100).toFixed(1)),
        rf: parseFloat((addNoise(modelAccuracies.rf, binCenter) * 100).toFixed(1)),
        svm: parseFloat((addNoise(modelAccuracies.svm, binCenter) * 100).toFixed(1)),
        xgboost: parseFloat((addNoise(modelAccuracies.xgboost, binCenter) * 100).toFixed(1)),
        knn: parseFloat((addNoise(modelAccuracies.knn, binCenter) * 100).toFixed(1)),
        mlp: parseFloat((addNoise(modelAccuracies.mlp, binCenter) * 100).toFixed(1)),
        ensemble: parseFloat((addNoise(modelAccuracies.ensemble, binCenter) * 100).toFixed(1)),
        perfect: parseFloat((binCenter * 100).toFixed(1)),
      });
    }
    
    return points;
  };

  // Calculate Expected Calibration Error (ECE) for each model
  const calculateECE = (modelKey: keyof Omit<CalibrationPoint, 'binCenter' | 'perfect'>) => {
    const calibrationData = generateCalibrationData();
    const n = calibrationData.length;
    let ece = 0;
    
    calibrationData.forEach(point => {
      const predicted = point[modelKey];
      const actual = point.perfect;
      if (predicted !== null) {
        ece += Math.abs((predicted as number) - actual);
      }
    });
    
    return (ece / n / 100).toFixed(3);
  };

  const calibrationData = generateCalibrationData();

  const modelColors = {
    rf: "hsl(var(--primary))",
    svm: "hsl(var(--secondary))",
    xgboost: "hsl(var(--accent))",
    knn: "hsl(var(--info))",
    mlp: "#f97316",
    ensemble: "#10b981",
  };

  const modelLabels: Record<string, string> = {
    rf: "Random Forest",
    svm: "SVM",
    xgboost: "XGBoost",
    knn: "KNN",
    mlp: "MLP",
    ensemble: "Ensemble",
  };

  return (
    <div className="space-y-6">
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            Calibration Curves (Reliability Diagrams)
            <Badge variant="outline" className="ml-2">Simulated</Badge>
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Calibration curves show how well predicted probabilities match actual outcomes. 
            A perfectly calibrated model follows the diagonal line.
          </p>
        </CardHeader>
        <CardContent>
          <div className="h-[500px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={calibrationData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                <XAxis 
                  dataKey="binCenter" 
                  label={{ value: 'Mean Predicted Probability (%)', position: 'bottom', offset: 0 }}
                  domain={[0, 100]}
                />
                <YAxis 
                  label={{ value: 'Fraction of Positives (%)', angle: -90, position: 'insideLeft' }}
                  domain={[0, 100]}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                  }}
                  formatter={(value: number, name: string) => [
                    `${value.toFixed(1)}%`,
                    modelLabels[name] || name
                  ]}
                />
                <Legend />
                
                {/* Perfect calibration line */}
                <Line 
                  type="linear" 
                  dataKey="perfect" 
                  stroke="#888" 
                  strokeDasharray="5 5" 
                  strokeWidth={2}
                  dot={false}
                  name="Perfect Calibration"
                />
                
                {/* Model calibration curves */}
                {Object.entries(modelColors).map(([key, color]) => (
                  <Line
                    key={key}
                    type="monotone"
                    dataKey={key}
                    stroke={color}
                    strokeWidth={2}
                    dot={{ fill: color, strokeWidth: 2, r: 4 }}
                    name={modelLabels[key]}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* ECE Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {Object.entries(modelLabels).map(([key, label]) => {
          const ece = calculateECE(key as keyof Omit<CalibrationPoint, 'binCenter' | 'perfect'>);
          const eceValue = parseFloat(ece);
          let eceClass = "text-green-500";
          if (eceValue > 0.1) eceClass = "text-red-500";
          else if (eceValue > 0.05) eceClass = "text-yellow-500";
          
          return (
            <Card key={key} className="bg-card border-border">
              <CardContent className="p-4 text-center">
                <p className="text-xs text-muted-foreground mb-1">{label}</p>
                <p className={`text-2xl font-bold ${eceClass}`}>{ece}</p>
                <p className="text-xs text-muted-foreground mt-1">ECE</p>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Interpretation Guide */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="text-lg">Interpretation Guide</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm text-muted-foreground">
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <h4 className="font-semibold text-foreground">Reading the Chart</h4>
              <ul className="list-disc list-inside space-y-1">
                <li><strong>Diagonal line:</strong> Perfect calibration</li>
                <li><strong>Above diagonal:</strong> Under-confident predictions</li>
                <li><strong>Below diagonal:</strong> Over-confident predictions</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold text-foreground">Expected Calibration Error (ECE)</h4>
              <ul className="list-disc list-inside space-y-1">
                <li><span className="text-green-500">ECE &lt; 0.05:</span> Well calibrated</li>
                <li><span className="text-yellow-500">ECE 0.05-0.10:</span> Moderately calibrated</li>
                <li><span className="text-red-500">ECE &gt; 0.10:</span> Poorly calibrated</li>
              </ul>
            </div>
          </div>
          <p>
            <strong>Why calibration matters:</strong> A model might have high accuracy but poor calibration, 
            meaning its probability predictions don't reflect true likelihoods. This is critical for 
            clinical decision-making where probability thresholds guide treatment decisions.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
