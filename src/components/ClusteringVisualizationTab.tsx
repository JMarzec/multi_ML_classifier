import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ZAxis } from "recharts";
import type { MLResults } from "@/types/ml-results";

interface ClusteringVisualizationTabProps {
  data: MLResults;
}

type MethodType = "pca" | "tsne" | "umap";

interface DataPoint {
  x: number;
  y: number;
  sampleId: string;
  actualClass: string;
  predictedClass: string;
  confidence: number;
  correct: boolean;
}

export function ClusteringVisualizationTab({ data }: ClusteringVisualizationTabProps) {
  const [method, setMethod] = useState<MethodType>("pca");
  const [colorBy, setColorBy] = useState<"actual" | "predicted" | "correct">("actual");

  // Check for exported data from R script
  const exportedPca = data.clustering?.pca?.points;
  const exportedTsne = data.clustering?.tsne?.points;
  const exportedUmap = data.clustering?.umap?.points;
  const exportedVariance = data.clustering?.pca?.variance_explained;

  // Determine which methods have real data
  const hasRealData = {
    pca: exportedPca && exportedPca.length > 0,
    tsne: exportedTsne && exportedTsne.length > 0,
    umap: exportedUmap && exportedUmap.length > 0,
  };

  // Build lookup map for ranking info
  const probBySampleId = new Map<string, { prob: number; pred: string; conf: number; correct: boolean; actual: string }>();
  (data.profile_ranking?.all_rankings || []).forEach((r) => {
    const sampleId = r.sample_id || r.sample_index?.toString();
    probBySampleId.set(sampleId, {
      prob: r.ensemble_probability,
      pred: r.predicted_class,
      conf: r.confidence,
      correct: r.correct,
      actual: r.actual_class,
    });
  });

  // Helper to convert exported points to DataPoint[]
  const convertExportedPoints = (points: typeof exportedPca): DataPoint[] => {
    if (!points) return [];
    return points.map((p) => {
      const r = probBySampleId.get(p.sample_id);
      return {
        x: Number(p.x.toFixed(2)),
        y: Number(p.y.toFixed(2)),
        sampleId: p.sample_id,
        actualClass: p.actual_class,
        predictedClass: r ? r.pred : "-",
        confidence: r?.conf ?? 0,
        correct: r?.correct ?? false,
      };
    });
  };

  const clusteringData: DataPoint[] = (() => {
    if (method === "pca" && hasRealData.pca) {
      return convertExportedPoints(exportedPca);
    }
    if (method === "tsne" && hasRealData.tsne) {
      return convertExportedPoints(exportedTsne);
    }
    if (method === "umap" && hasRealData.umap) {
      return convertExportedPoints(exportedUmap);
    }
    // Fallback to simulated data
    return [];
  })();

  const getScatterData = () => {
    if (colorBy === "correct") return [
      { data: clusteringData.filter(d => d.correct), name: "Correct", fill: "hsl(var(--success))" },
      { data: clusteringData.filter(d => !d.correct), name: "Misclassified", fill: "hsl(var(--destructive))" },
    ];
    const classes = [...new Set(clusteringData.map(d => colorBy === "actual" ? d.actualClass : d.predictedClass))];
    const colors = ["hsl(var(--primary))", "hsl(var(--secondary))", "hsl(var(--accent))"];
    return classes.map((label, i) => ({
      data: clusteringData.filter(d => (colorBy === "actual" ? d.actualClass : d.predictedClass) === label),
      name: `${label} (${colorBy === "actual" ? "Actual" : "Predicted"})`, fill: colors[i % colors.length],
    }));
  };

  const scatterGroups = getScatterData();

  const getAxisLabel = () => {
    switch (method) {
      case "pca": return { x: "PC1", y: "PC2" };
      case "tsne": return { x: "t-SNE 1", y: "t-SNE 2" };
      case "umap": return { x: "UMAP 1", y: "UMAP 2" };
    }
  };

  const axisLabel = getAxisLabel();

  const varianceExplained = method === "pca" && exportedVariance ? {
    pc1: (exportedVariance.pc1 * 100).toFixed(1),
    pc2: (exportedVariance.pc2 * 100).toFixed(1),
  } : null;

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex flex-wrap gap-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Reduction Method</label>
          <div className="flex gap-2">
            {(["pca", "tsne", "umap"] as MethodType[]).map((m) => (
              <Button
                key={m}
                variant={method === m ? "default" : "outline"}
                size="sm"
                onClick={() => setMethod(m)}
              >
                {m.toUpperCase()}
              </Button>
            ))}
          </div>
        </div>
        
        <div className="space-y-2">
          <label className="text-sm font-medium">Color By</label>
          <div className="flex gap-2">
            {([
              { key: "actual", label: "Actual Class" },
              { key: "predicted", label: "Predicted Class" },
              { key: "correct", label: "Correctness" },
            ] as const).map(({ key, label }) => (
              <Button
                key={key}
                variant={colorBy === key ? "default" : "outline"}
                size="sm"
                onClick={() => setColorBy(key)}
              >
                {label}
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Visualization */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            Sample Clustering - {method.toUpperCase()}
            <Badge variant="outline" className="ml-2">
              {hasRealData[method] ? "From analysis" : "Unavailable"}
            </Badge>
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Visualization of how samples cluster in reduced dimensional space using selected features.
          </p>
        </CardHeader>
        <CardContent>
          {clusteringData.length === 0 ? <p className="py-16 text-center text-muted-foreground">No {method.toUpperCase()} coordinates were exported for this run.</p> : <div className="h-[500px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                <XAxis 
                  type="number" 
                  dataKey="x" 
                  name={axisLabel.x}
                  label={{ value: axisLabel.x, position: 'bottom', offset: 0 }}
                />
                <YAxis 
                  type="number" 
                  dataKey="y" 
                  name={axisLabel.y}
                  label={{ value: axisLabel.y, angle: -90, position: 'insideLeft' }}
                />
                <ZAxis type="number" dataKey="confidence" range={[50, 200]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                  }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload as DataPoint;
                      return (
                        <div className="bg-card p-3 rounded-lg border border-border shadow-lg">
                          <p className="font-semibold">{data.sampleId}</p>
                          <p className="text-sm">Actual: {data.actualClass}</p>
                          <p className="text-sm">Predicted: {data.predictedClass}</p>
                          <p className="text-sm">Confidence: {(data.confidence * 100).toFixed(1)}%</p>
                          <p className={`text-sm font-medium ${data.correct ? 'text-success' : 'text-destructive'}`}>
                            {data.correct ? '✓ Correct' : '✗ Misclassified'}
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend />
                
                {scatterGroups.map((group, idx) => (
                  <Scatter
                    key={idx}
                    name={group.name}
                    data={group.data}
                    fill={group.fill}
                    opacity={0.7}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>}
        </CardContent>
      </Card>

      {/* Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-card border-border">
          <CardContent className="p-4 text-center">
            <p className="text-xs text-muted-foreground mb-1">Silhouette Score</p>
            <p className="text-2xl font-bold text-primary">N/A</p>
          </CardContent>
        </Card>
        
        {varianceExplained && (
          <>
            <Card className="bg-card border-border">
              <CardContent className="p-4 text-center">
                <p className="text-xs text-muted-foreground mb-1">PC1 Variance</p>
                <p className="text-2xl font-bold text-secondary">{varianceExplained.pc1}%</p>
              </CardContent>
            </Card>
            <Card className="bg-card border-border">
              <CardContent className="p-4 text-center">
                <p className="text-xs text-muted-foreground mb-1">PC2 Variance</p>
                <p className="text-2xl font-bold text-accent">{varianceExplained.pc2}%</p>
              </CardContent>
            </Card>
          </>
        )}
        
        <Card className="bg-card border-border">
          <CardContent className="p-4 text-center">
            <p className="text-xs text-muted-foreground mb-1">Samples</p>
            <p className="text-2xl font-bold text-info">{clusteringData.length}</p>
          </CardContent>
        </Card>
        
        <Card className="bg-card border-border">
          <CardContent className="p-4 text-center">
            <p className="text-xs text-muted-foreground mb-1">Features Used</p>
            <p className="text-2xl font-bold text-foreground">{data.selected_features?.length || 0}</p>
          </CardContent>
        </Card>
      </div>

      {/* Interpretation Guide */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="text-lg">Interpretation Guide</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm text-muted-foreground">
          <div className="grid md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <h4 className="font-semibold text-foreground">PCA</h4>
              <p>Linear dimensionality reduction that preserves global structure and variance. Best for identifying major patterns.</p>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold text-foreground">t-SNE</h4>
              <p>Non-linear method that preserves local structure. Good for visualizing clusters but distances are not meaningful.</p>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold text-foreground">UMAP</h4>
              <p>Preserves both local and global structure. Generally faster than t-SNE with comparable or better results.</p>
            </div>
          </div>
          <p>
            <strong>Good separation:</strong> Clear visual separation between classes indicates that the selected features 
            effectively distinguish between diagnostic groups, supporting model reliability.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
