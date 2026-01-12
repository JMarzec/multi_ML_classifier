import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { MLResults } from "@/types/ml-results";

interface CVFoldVisualizationTabProps {
  data: MLResults;
}

type MetricType = "accuracy" | "auroc" | "sensitivity" | "specificity" | "f1_score";

interface FoldMetric {
  fold: string;
  rf: number;
  svm: number;
  xgboost: number;
  knn: number;
  mlp: number;
  ensemble: number;
}

interface ModelStats {
  model: string;
  mean: number;
  sd: number;
  min: number;
  max: number;
  q25: number;
  median: number;
  q75: number;
  cv: number; // coefficient of variation
}

export function CVFoldVisualizationTab({ data }: CVFoldVisualizationTabProps) {
  const [metric, setMetric] = useState<MetricType>("auroc");
  const [viewMode, setViewMode] = useState<"bar" | "box" | "table">("bar");

  const config = data.metadata.config;
  const totalFolds = config.n_folds * config.n_repeats;

  // Generate simulated fold-level data based on overall performance
  const generateFoldData = (metricType: MetricType): FoldMetric[] => {
    const folds: FoldMetric[] = [];
    
    const getModelMetric = (modelKey: string) => {
      const perf = data.model_performance[modelKey as keyof typeof data.model_performance];
      return perf?.[metricType]?.mean || 0.75;
    };
    
    const getModelSD = (modelKey: string) => {
      const perf = data.model_performance[modelKey as keyof typeof data.model_performance];
      return perf?.[metricType]?.sd || 0.05;
    };
    
    for (let i = 0; i < totalFolds; i++) {
      const repeatNum = Math.floor(i / config.n_folds) + 1;
      const foldNum = (i % config.n_folds) + 1;
      
      // Generate values with some variation around the mean
      const generateValue = (modelKey: string) => {
        const mean = getModelMetric(modelKey);
        const sd = getModelSD(modelKey);
        // Simulate variation within reported SD
        const noise = (Math.random() - 0.5) * sd * 2;
        return Math.max(0, Math.min(1, mean + noise));
      };
      
      folds.push({
        fold: `R${repeatNum}F${foldNum}`,
        rf: parseFloat((generateValue("rf") * 100).toFixed(1)),
        svm: parseFloat((generateValue("svm") * 100).toFixed(1)),
        xgboost: parseFloat((generateValue("xgboost") * 100).toFixed(1)),
        knn: parseFloat((generateValue("knn") * 100).toFixed(1)),
        mlp: parseFloat((generateValue("mlp") * 100).toFixed(1)),
        ensemble: parseFloat((generateValue("soft_vote") * 100).toFixed(1)),
      });
    }
    
    return folds;
  };

  const foldData = generateFoldData(metric);

  // Calculate statistics for each model
  const calculateModelStats = (modelKey: keyof Omit<FoldMetric, 'fold'>): ModelStats => {
    const values = foldData.map(f => f[modelKey]).sort((a, b) => a - b);
    const n = values.length;
    const mean = values.reduce((a, b) => a + b, 0) / n;
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
    const sd = Math.sqrt(variance);
    
    return {
      model: modelKey.toUpperCase(),
      mean: parseFloat(mean.toFixed(1)),
      sd: parseFloat(sd.toFixed(2)),
      min: values[0],
      max: values[n - 1],
      q25: values[Math.floor(n * 0.25)],
      median: values[Math.floor(n * 0.5)],
      q75: values[Math.floor(n * 0.75)],
      cv: parseFloat(((sd / mean) * 100).toFixed(1)),
    };
  };

  const modelStats: ModelStats[] = [
    calculateModelStats("rf"),
    calculateModelStats("svm"),
    calculateModelStats("xgboost"),
    calculateModelStats("knn"),
    calculateModelStats("mlp"),
    calculateModelStats("ensemble"),
  ];

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

  // Prepare data for box plot visualization
  const boxPlotData = Object.keys(modelColors).map(modelKey => {
    const stats = modelStats.find(s => s.model === modelKey.toUpperCase())!;
    return {
      name: modelLabels[modelKey],
      min: stats.min,
      q25: stats.q25,
      median: stats.median,
      q75: stats.q75,
      max: stats.max,
      fill: modelColors[modelKey as keyof typeof modelColors],
    };
  });

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex flex-wrap gap-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Metric</label>
          <div className="flex gap-2 flex-wrap">
            {(["auroc", "accuracy", "sensitivity", "specificity", "f1_score"] as MetricType[]).map((m) => (
              <Button
                key={m}
                variant={metric === m ? "default" : "outline"}
                size="sm"
                onClick={() => setMetric(m)}
              >
                {m.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
              </Button>
            ))}
          </div>
        </div>
        
        <div className="space-y-2">
          <label className="text-sm font-medium">View</label>
          <div className="flex gap-2">
            {([
              { key: "bar", label: "Fold Chart" },
              { key: "box", label: "Distribution" },
              { key: "table", label: "Statistics" },
            ] as const).map(({ key, label }) => (
              <Button
                key={key}
                variant={viewMode === key ? "default" : "outline"}
                size="sm"
                onClick={() => setViewMode(key)}
              >
                {label}
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* Fold-Level Bar Chart */}
      {viewMode === "bar" && (
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {metric.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())} by CV Fold
              <Badge variant="outline" className="ml-2">Simulated</Badge>
            </CardTitle>
            <p className="text-sm text-muted-foreground">
              Performance of each model across {totalFolds} cross-validation folds 
              ({config.n_folds}-fold CV × {config.n_repeats} repeats)
            </p>
          </CardHeader>
          <CardContent>
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={foldData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis 
                    dataKey="fold" 
                    angle={-45}
                    textAnchor="end"
                    height={60}
                    fontSize={10}
                  />
                  <YAxis 
                    domain={[0, 100]}
                    label={{ value: `${metric.toUpperCase()} (%)`, angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                    }}
                  />
                  <Legend wrapperStyle={{ paddingTop: 20 }} />
                  
                  {Object.entries(modelColors).map(([key, color]) => (
                    <Bar
                      key={key}
                      dataKey={key}
                      name={modelLabels[key]}
                      fill={color}
                      opacity={0.8}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Box Plot Distribution */}
      {viewMode === "box" && (
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {metric.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())} Distribution
              <Badge variant="outline" className="ml-2">Simulated</Badge>
            </CardTitle>
            <p className="text-sm text-muted-foreground">
              Distribution of {metric} values across all CV folds for each model
            </p>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              {boxPlotData.map((model) => (
                <div key={model.name} className="bg-muted/30 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-center mb-4">{model.name}</h4>
                  <div className="relative h-48 flex items-center justify-center">
                    <div className="relative w-16 h-full">
                      {/* Whiskers */}
                      <div 
                        className="absolute left-1/2 w-px bg-foreground/50"
                        style={{
                          top: `${100 - model.max}%`,
                          height: `${model.max - model.min}%`,
                          transform: 'translateX(-50%)',
                        }}
                      />
                      
                      {/* Box */}
                      <div 
                        className="absolute left-1/2 w-10 rounded border-2"
                        style={{
                          backgroundColor: model.fill,
                          opacity: 0.7,
                          top: `${100 - model.q75}%`,
                          height: `${model.q75 - model.q25}%`,
                          transform: 'translateX(-50%)',
                          borderColor: model.fill,
                        }}
                      />
                      
                      {/* Median line */}
                      <div 
                        className="absolute left-1/2 w-12 h-0.5 bg-white"
                        style={{
                          top: `${100 - model.median}%`,
                          transform: 'translateX(-50%)',
                        }}
                      />
                      
                      {/* Min/Max caps */}
                      <div 
                        className="absolute left-1/2 w-6 h-px bg-foreground/50"
                        style={{
                          top: `${100 - model.max}%`,
                          transform: 'translateX(-50%)',
                        }}
                      />
                      <div 
                        className="absolute left-1/2 w-6 h-px bg-foreground/50"
                        style={{
                          top: `${100 - model.min}%`,
                          transform: 'translateX(-50%)',
                        }}
                      />
                    </div>
                  </div>
                  <div className="text-center text-xs text-muted-foreground mt-2 space-y-1">
                    <p>Median: {model.median.toFixed(1)}%</p>
                    <p>IQR: {(model.q75 - model.q25).toFixed(1)}%</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Statistics Table */}
      {viewMode === "table" && (
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {metric.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())} Statistics
              <Badge variant="outline" className="ml-2">Simulated</Badge>
            </CardTitle>
            <p className="text-sm text-muted-foreground">
              Detailed statistics for {metric} across all CV folds
            </p>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead className="text-right">Mean (%)</TableHead>
                  <TableHead className="text-right">SD</TableHead>
                  <TableHead className="text-right">CV (%)</TableHead>
                  <TableHead className="text-right">Min</TableHead>
                  <TableHead className="text-right">Q25</TableHead>
                  <TableHead className="text-right">Median</TableHead>
                  <TableHead className="text-right">Q75</TableHead>
                  <TableHead className="text-right">Max</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {modelStats.map((stats) => (
                  <TableRow key={stats.model}>
                    <TableCell className="font-medium">
                      {modelLabels[stats.model.toLowerCase()] || stats.model}
                    </TableCell>
                    <TableCell className="text-right font-mono">{stats.mean}</TableCell>
                    <TableCell className="text-right font-mono">{stats.sd}</TableCell>
                    <TableCell className="text-right font-mono">
                      <span className={stats.cv > 10 ? 'text-yellow-500' : 'text-green-500'}>
                        {stats.cv}
                      </span>
                    </TableCell>
                    <TableCell className="text-right font-mono">{stats.min}</TableCell>
                    <TableCell className="text-right font-mono">{stats.q25}</TableCell>
                    <TableCell className="text-right font-mono">{stats.median}</TableCell>
                    <TableCell className="text-right font-mono">{stats.q75}</TableCell>
                    <TableCell className="text-right font-mono">{stats.max}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* Stability Indicators */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {modelStats.map((stats) => {
          const stability = stats.cv < 5 ? "Excellent" : stats.cv < 10 ? "Good" : stats.cv < 15 ? "Fair" : "Poor";
          const stabilityColor = stats.cv < 5 ? "text-green-500" : stats.cv < 10 ? "text-blue-500" : stats.cv < 15 ? "text-yellow-500" : "text-red-500";
          
          return (
            <Card key={stats.model} className="bg-card border-border">
              <CardContent className="p-4 text-center">
                <p className="text-xs text-muted-foreground mb-1">{modelLabels[stats.model.toLowerCase()] || stats.model}</p>
                <p className={`text-lg font-bold ${stabilityColor}`}>{stability}</p>
                <p className="text-xs text-muted-foreground">CV: {stats.cv}%</p>
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
              <h4 className="font-semibold text-foreground">Coefficient of Variation (CV)</h4>
              <ul className="list-disc list-inside space-y-1">
                <li><span className="text-green-500">CV &lt; 5%:</span> Excellent stability</li>
                <li><span className="text-blue-500">CV 5-10%:</span> Good stability</li>
                <li><span className="text-yellow-500">CV 10-15%:</span> Fair stability</li>
                <li><span className="text-red-500">CV &gt; 15%:</span> Poor stability</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold text-foreground">Why Fold Variance Matters</h4>
              <p>
                High variance across folds indicates that model performance depends heavily on which 
                samples are in the training/test split. This suggests potential overfitting or 
                sensitivity to specific samples.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
