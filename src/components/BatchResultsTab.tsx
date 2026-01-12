import { useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { Upload, Layers, BarChart3, X, CheckCircle, AlertCircle } from "lucide-react";
import { toast } from "sonner";

interface BatchDataset {
  name: string;
  status: "success" | "failed";
  summary?: {
    rf?: { auroc?: { mean: number } };
    svm?: { auroc?: { mean: number } };
    xgboost?: { auroc?: { mean: number } };
    knn?: { auroc?: { mean: number } };
    mlp?: { auroc?: { mean: number } };
    soft_vote?: { auroc?: { mean: number } };
  };
  selected_features?: string[];
  error?: string;
}

interface BatchResults {
  metadata: {
    generated_at: string;
    batch_mode: boolean;
    n_datasets: number;
  };
  datasets: Record<string, BatchDataset>;
  summary?: Array<{
    dataset: string;
    status: string;
    rf_auroc: number | null;
    soft_vote_auroc: number | null;
    n_features: number | null;
  }>;
}

export function BatchResultsTab() {
  const [batchData, setBatchData] = useState<BatchResults | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const json = JSON.parse(e.target?.result as string);
        if (!json.metadata?.batch_mode) {
          toast.error("This doesn't appear to be a batch results file");
          return;
        }
        setBatchData(json);
        toast.success("Batch results loaded successfully");
      } catch {
        toast.error("Failed to parse batch results file");
      }
    };
    reader.readAsText(file);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file?.name.endsWith(".json")) {
      handleFile(file);
    } else {
      toast.error("Please drop a JSON file");
    }
  }, [handleFile]);

  if (!batchData) {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Layers className="w-5 h-5 text-primary" />
              Batch Results Viewer
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className={`border-2 border-dashed rounded-xl p-12 text-center transition-colors ${
                isDragging ? "border-primary bg-primary/5" : "border-border"
              }`}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
            >
              <Upload className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">Upload Batch Results</h3>
              <p className="text-muted-foreground mb-4">
                Drop your batch_results.json file here or click to browse
              </p>
              <input
                type="file"
                accept=".json"
                className="hidden"
                id="batch-upload"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFile(file);
                }}
              />
              <Button asChild>
                <label htmlFor="batch-upload" className="cursor-pointer">
                  <Upload className="w-4 h-4 mr-2" />
                  Select File
                </label>
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const datasets = Object.entries(batchData.datasets);
  const successfulDatasets = datasets.filter(([, d]) => d.status === "success" || !d.error);

  // Prepare comparison chart data
  const comparisonData = successfulDatasets.map(([name, dataset]) => ({
    name: name.length > 15 ? name.slice(0, 15) + "..." : name,
    fullName: name,
    rf: (dataset.summary?.rf?.auroc?.mean || 0) * 100,
    svm: (dataset.summary?.svm?.auroc?.mean || 0) * 100,
    xgboost: (dataset.summary?.xgboost?.auroc?.mean || 0) * 100,
    knn: (dataset.summary?.knn?.auroc?.mean || 0) * 100,
    mlp: (dataset.summary?.mlp?.auroc?.mean || 0) * 100,
    soft_vote: (dataset.summary?.soft_vote?.auroc?.mean || 0) * 100,
  }));

  // Prepare radar chart data for model comparison across datasets
  const radarData = ["rf", "svm", "xgboost", "knn", "mlp", "soft_vote"].map((model) => {
    const dataPoint: Record<string, number | string> = { model: model.toUpperCase() };
    successfulDatasets.forEach(([name, dataset]) => {
      const auroc = (dataset.summary as Record<string, { auroc?: { mean: number } }>)?.[model]?.auroc?.mean || 0;
      dataPoint[name] = auroc * 100;
    });
    return dataPoint;
  });

  // Find common features across datasets
  const featureSets = successfulDatasets
    .filter(([, d]) => d.selected_features)
    .map(([name, d]) => ({ name, features: new Set(d.selected_features || []) }));
  
  const commonFeatures = featureSets.length > 0
    ? [...featureSets[0].features].filter((f) =>
        featureSets.every((set) => set.features.has(f))
      )
    : [];

  const colors = [
    "hsl(var(--primary))",
    "hsl(var(--secondary))",
    "hsl(var(--accent))",
    "hsl(var(--warning))",
    "hsl(var(--info))",
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Batch Analysis Results</h2>
          <p className="text-muted-foreground">
            {batchData.metadata.n_datasets} datasets analyzed on{" "}
            {new Date(batchData.metadata.generated_at).toLocaleDateString()}
          </p>
        </div>
        <Button variant="outline" onClick={() => setBatchData(null)}>
          <X className="w-4 h-4 mr-2" />
          Clear
        </Button>
      </div>

      {/* Dataset Status Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {datasets.map(([name, dataset]) => (
          <Card key={name} className={dataset.error ? "border-destructive/50" : "border-success/50"}>
            <CardContent className="pt-4">
              <div className="flex items-start justify-between">
                <div>
                  <h4 className="font-medium text-sm truncate" title={name}>{name}</h4>
                  <p className="text-xs text-muted-foreground mt-1">
                    {dataset.selected_features?.length || 0} features
                  </p>
                </div>
                {dataset.error ? (
                  <AlertCircle className="w-5 h-5 text-destructive shrink-0" />
                ) : (
                  <CheckCircle className="w-5 h-5 text-success shrink-0" />
                )}
              </div>
              {!dataset.error && dataset.summary?.soft_vote?.auroc && (
                <div className="mt-3 text-center">
                  <span className="text-2xl font-bold text-primary">
                    {(dataset.summary.soft_vote.auroc.mean * 100).toFixed(1)}%
                  </span>
                  <p className="text-xs text-muted-foreground">Ensemble AUROC</p>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* AUROC Comparison Bar Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-primary" />
            AUROC Comparison Across Datasets
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis 
                  dataKey="name" 
                  angle={-45} 
                  textAnchor="end" 
                  height={80}
                  className="text-muted-foreground"
                />
                <YAxis domain={[0, 100]} label={{ value: 'AUROC (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--popover))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                  formatter={(value: number) => `${value.toFixed(1)}%`}
                />
                <Legend />
                <Bar dataKey="rf" name="Random Forest" fill="hsl(var(--primary))" />
                <Bar dataKey="svm" name="SVM" fill="hsl(var(--secondary))" />
                <Bar dataKey="xgboost" name="XGBoost" fill="hsl(var(--accent))" />
                <Bar dataKey="soft_vote" name="Soft Vote" fill="hsl(var(--success))" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Radar Chart for Model Comparison */}
      {successfulDatasets.length <= 5 && successfulDatasets.length > 1 && (
        <Card>
          <CardHeader>
            <CardTitle>Model Performance by Dataset</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid className="stroke-border" />
                  <PolarAngleAxis dataKey="model" className="text-muted-foreground" />
                  <PolarRadiusAxis domain={[0, 100]} />
                  {successfulDatasets.map(([name], idx) => (
                    <Radar
                      key={name}
                      name={name}
                      dataKey={name}
                      stroke={colors[idx % colors.length]}
                      fill={colors[idx % colors.length]}
                      fillOpacity={0.2}
                    />
                  ))}
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Common Features */}
      {commonFeatures.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Common Features Across All Datasets ({commonFeatures.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {commonFeatures.slice(0, 20).map((feature) => (
                <span
                  key={feature}
                  className="px-3 py-1 bg-success/10 text-success rounded-full text-sm font-mono"
                >
                  {feature}
                </span>
              ))}
              {commonFeatures.length > 20 && (
                <span className="px-3 py-1 bg-muted text-muted-foreground rounded-full text-sm">
                  +{commonFeatures.length - 20} more
                </span>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
