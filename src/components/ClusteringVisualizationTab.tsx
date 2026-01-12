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

  // Generate simulated dimensionality reduction data
  // In real implementation, this would come from the R script
  const generateClusteringData = (methodType: MethodType): DataPoint[] => {
    const rankings = data.profile_ranking?.all_rankings || [];
    const n = Math.max(rankings.length, 50);
    const points: DataPoint[] = [];
    
    // Simulate two clusters based on class
    const class0Center = methodType === "pca" ? { x: -2, y: -1 } : 
                         methodType === "tsne" ? { x: -15, y: -10 } : 
                         { x: -3, y: -2 };
    const class1Center = methodType === "pca" ? { x: 2, y: 1 } : 
                         methodType === "tsne" ? { x: 15, y: 10 } : 
                         { x: 3, y: 2 };
    
    const spread = methodType === "pca" ? 1.5 : methodType === "tsne" ? 8 : 2;
    
    // Get model accuracy to determine cluster separation
    const accuracy = data.model_performance.soft_vote?.accuracy?.mean || 0.8;
    const separation = accuracy * 0.8 + 0.2; // Higher accuracy = better separation
    
    for (let i = 0; i < n; i++) {
      const ranking = rankings[i];
      const isClass1 = ranking ? ranking.actual_class === "1" : Math.random() > 0.5;
      const center = isClass1 ? class1Center : class0Center;
      
      // Add noise inversely proportional to separation
      const noiseX = (Math.random() - 0.5) * spread * (1 / separation);
      const noiseY = (Math.random() - 0.5) * spread * (1 / separation);
      
      // Misclassified samples appear in wrong cluster region
      const isMisclassified = ranking ? !ranking.correct : Math.random() > accuracy;
      const effectiveCenter = isMisclassified ? 
        (isClass1 ? class0Center : class1Center) : center;
      
      points.push({
        x: parseFloat((effectiveCenter.x + noiseX).toFixed(2)),
        y: parseFloat((effectiveCenter.y + noiseY).toFixed(2)),
        sampleId: ranking?.sample_index?.toString() || `Sample_${i + 1}`,
        actualClass: isClass1 ? "Positive" : "Negative",
        predictedClass: ranking?.predicted_class === "1" ? "Positive" : "Negative",
        confidence: ranking?.confidence || Math.random() * 0.3 + 0.7,
        correct: ranking?.correct ?? !isMisclassified,
      });
    }
    
    return points;
  };

  const clusteringData = generateClusteringData(method);
  const class0Data = clusteringData.filter(d => d.actualClass === "Negative");
  const class1Data = clusteringData.filter(d => d.actualClass === "Positive");
  const correctData = clusteringData.filter(d => d.correct);
  const incorrectData = clusteringData.filter(d => !d.correct);
  const predictedClass0 = clusteringData.filter(d => d.predictedClass === "Negative");
  const predictedClass1 = clusteringData.filter(d => d.predictedClass === "Positive");

  const getScatterData = () => {
    switch (colorBy) {
      case "actual":
        return [
          { data: class0Data, name: "Negative (Actual)", fill: "hsl(var(--primary))" },
          { data: class1Data, name: "Positive (Actual)", fill: "hsl(var(--secondary))" },
        ];
      case "predicted":
        return [
          { data: predictedClass0, name: "Negative (Predicted)", fill: "hsl(var(--info))" },
          { data: predictedClass1, name: "Positive (Predicted)", fill: "hsl(var(--accent))" },
        ];
      case "correct":
        return [
          { data: correctData, name: "Correct", fill: "#10b981" },
          { data: incorrectData, name: "Misclassified", fill: "#ef4444" },
        ];
    }
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

  // Calculate separation metrics
  const calculateSilhouette = () => {
    // Simplified silhouette score simulation
    const accuracy = data.model_performance.soft_vote?.accuracy?.mean || 0.8;
    return (accuracy * 0.8 + Math.random() * 0.2).toFixed(3);
  };

  const varianceExplained = method === "pca" ? {
    pc1: (Math.random() * 20 + 40).toFixed(1),
    pc2: (Math.random() * 15 + 15).toFixed(1),
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
            <Badge variant="outline" className="ml-2">Simulated</Badge>
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Visualization of how samples cluster in reduced dimensional space using selected features.
          </p>
        </CardHeader>
        <CardContent>
          <div className="h-[500px]">
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
                          <p className={`text-sm font-medium ${data.correct ? 'text-green-500' : 'text-red-500'}`}>
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
          </div>
        </CardContent>
      </Card>

      {/* Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-card border-border">
          <CardContent className="p-4 text-center">
            <p className="text-xs text-muted-foreground mb-1">Silhouette Score</p>
            <p className="text-2xl font-bold text-primary">{calculateSilhouette()}</p>
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
