import { useState } from "react";
import { FileUploader } from "@/components/FileUploader";
import { Dashboard } from "@/components/Dashboard";
import { Button } from "@/components/ui/button";
import { Brain, Download, Github, FileCode2, Sparkles } from "lucide-react";
import type { MLResults } from "@/types/ml-results";

const Index = () => {
  const [data, setData] = useState<MLResults | null>(null);

  if (data) {
    return <Dashboard data={data} onReset={() => setData(null)} />;
  }

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Background gradient effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-primary/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-secondary/5 rounded-full blur-[100px]" />
      </div>

      <div className="relative z-10 container mx-auto px-4 py-16">
        {/* Hero Section */}
        <div className="text-center max-w-4xl mx-auto mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 rounded-full mb-6 border border-primary/20">
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="text-sm text-primary font-medium">IntelliGenes-Style ML Classification</span>
          </div>
          
          <h1 className="text-5xl md:text-6xl font-bold mb-6">
            <span className="gradient-text">Multi-Method ML</span>
            <br />
            <span className="text-foreground">Diagnostic Classifier</span>
          </h1>
          
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
            Ensemble machine learning with Random Forest, SVM, XGBoost, KNN, and MLP. 
            Includes permutation testing, feature selection, and profile ranking for robust diagnostic predictions.
          </p>

          <div className="flex flex-wrap justify-center gap-4 mb-12">
            <a href="/intelligenes_ml_classifier.R" download>
              <Button size="lg" className="gap-2 glow-primary">
                <Download className="w-5 h-5" />
                Download R Script
              </Button>
            </a>
            <a 
              href="https://github.com/drzeeshanahmed/intelligenes" 
              target="_blank" 
              rel="noopener noreferrer"
            >
              <Button variant="outline" size="lg" className="gap-2">
                <Github className="w-5 h-5" />
                View IntelliGenes
              </Button>
            </a>
          </div>
        </div>

        {/* Upload Section */}
        <div className="max-w-2xl mx-auto mb-16">
          <div className="text-center mb-6">
            <h2 className="text-2xl font-semibold mb-2">Upload Results</h2>
            <p className="text-muted-foreground">
              Upload the JSON output from the R script to visualize your results
            </p>
          </div>
          <FileUploader onDataLoaded={setData} />
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto mb-16">
          <div className="bg-card/50 backdrop-blur-sm rounded-xl p-6 border border-border hover:border-primary/30 transition-all group">
            <div className="w-12 h-12 rounded-lg bg-chart-rf/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <Brain className="w-6 h-6 text-chart-rf" />
            </div>
            <h3 className="text-lg font-semibold mb-2">5 ML Methods</h3>
            <p className="text-sm text-muted-foreground">
              Random Forest, SVM, XGBoost, K-Nearest Neighbors, and Multi-Layer Perceptron with soft/hard voting ensembles.
            </p>
          </div>

          <div className="bg-card/50 backdrop-blur-sm rounded-xl p-6 border border-border hover:border-primary/30 transition-all group">
            <div className="w-12 h-12 rounded-lg bg-chart-svm/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <FileCode2 className="w-6 h-6 text-chart-svm" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Feature Selection</h3>
            <p className="text-sm text-muted-foreground">
              Forward selection, backward elimination, and stepwise selection for optimal feature subset identification.
            </p>
          </div>

          <div className="bg-card/50 backdrop-blur-sm rounded-xl p-6 border border-border hover:border-primary/30 transition-all group">
            <div className="w-12 h-12 rounded-lg bg-chart-xgb/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <Sparkles className="w-6 h-6 text-chart-xgb" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Permutation Testing</h3>
            <p className="text-sm text-muted-foreground">
              Validates model robustness by comparing performance against randomly shuffled labels (Li et al. 2022).
            </p>
          </div>
        </div>

        {/* Workflow Steps */}
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl font-semibold text-center mb-8">Workflow</h2>
          <div className="grid md:grid-cols-4 gap-4">
            {[
              { step: 1, title: "Download", desc: "Get the R script" },
              { step: 2, title: "Prepare", desc: "Format your data as CSV" },
              { step: 3, title: "Run", desc: "Execute the R pipeline" },
              { step: 4, title: "Visualize", desc: "Upload JSON results here" },
            ].map(({ step, title, desc }) => (
              <div key={step} className="relative">
                <div className="bg-card rounded-xl p-5 border border-border text-center">
                  <div className="w-10 h-10 rounded-full bg-primary/20 text-primary font-bold flex items-center justify-center mx-auto mb-3">
                    {step}
                  </div>
                  <h3 className="font-semibold mb-1">{title}</h3>
                  <p className="text-sm text-muted-foreground">{desc}</p>
                </div>
                {step < 4 && (
                  <div className="hidden md:block absolute top-1/2 -right-2 w-4 h-0.5 bg-border" />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-20 pt-8 border-t border-border text-center text-sm text-muted-foreground">
          <p>
            Inspired by{" "}
            <a 
              href="https://academic.oup.com/bioinformatics/article/39/12/btad755/7473370" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              IntelliGenes
            </a>
            {" "}and{" "}
            <a 
              href="https://github.com/CoLAB-AccelBio/molecular-classification-analysis" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              Molecular Classification Analysis
            </a>
          </p>
        </footer>
      </div>
    </div>
  );
};

export default Index;
