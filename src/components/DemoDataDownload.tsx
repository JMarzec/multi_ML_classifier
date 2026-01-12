import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Download, FileSpreadsheet, FileText, Info } from "lucide-react";
import { toast } from "sonner";

export function DemoDataDownload() {
  const downloadFile = (filename: string, description: string) => {
    const link = document.createElement("a");
    link.href = `/demo_data/${filename}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    toast.success(`Downloading ${description}`);
  };

  return (
    <Card className="bg-gradient-to-br from-primary/5 to-secondary/5 border-primary/20">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Download className="w-5 h-5 text-primary" />
          Demo Input Files
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="bg-background/50 rounded-lg p-4 flex items-start gap-3">
          <Info className="w-5 h-5 text-muted-foreground mt-0.5 shrink-0" />
          <div className="text-sm text-muted-foreground">
            <p>
              Download these sample input files to test the R script. The data is from 
              GEO dataset GSE62254 (gastric cancer EMT subtypes) with the expression matrix 
              containing 2121 features across 39 samples.
            </p>
          </div>
        </div>

        <div className="grid sm:grid-cols-2 gap-4">
          <div className="bg-card rounded-lg p-4 border border-border">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                <FileSpreadsheet className="w-5 h-5 text-primary" />
              </div>
              <div>
                <h4 className="font-medium">Expression Matrix</h4>
                <p className="text-xs text-muted-foreground">expression_matrix.txt</p>
              </div>
            </div>
            <p className="text-sm text-muted-foreground mb-3">
              Tab-delimited file with features as rows and samples as columns. 
              Column names are sample IDs.
            </p>
            <Button 
              size="sm" 
              className="w-full"
              onClick={() => downloadFile("expression_matrix.txt", "expression matrix")}
            >
              <Download className="w-4 h-4 mr-2" />
              Download
            </Button>
          </div>

          <div className="bg-card rounded-lg p-4 border border-border">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-lg bg-secondary/10 flex items-center justify-center">
                <FileText className="w-5 h-5 text-secondary" />
              </div>
              <div>
                <h4 className="font-medium">Sample Annotation</h4>
                <p className="text-xs text-muted-foreground">sample_annotation.txt</p>
              </div>
            </div>
            <p className="text-sm text-muted-foreground mb-3">
              Tab-delimited file with sample metadata. First column contains sample IDs 
              matching the expression matrix.
            </p>
            <Button 
              size="sm" 
              variant="secondary"
              className="w-full"
              onClick={() => downloadFile("sample_annotation.txt", "sample annotation")}
            >
              <Download className="w-4 h-4 mr-2" />
              Download
            </Button>
          </div>
        </div>

        <div className="text-xs text-muted-foreground mt-4">
          <p className="font-medium mb-1">R Script Configuration for Demo:</p>
          <pre className="bg-muted/50 p-3 rounded-lg overflow-x-auto font-mono">
{`config <- list(
  expression_matrix_file = "expression_matrix.txt",
  annotation_file = "sample_annotation.txt",
  target_variable = "EMT_subtype",  # Binary: MP vs Non-MP
  analysis_mode = "fast"  # Use "full" for production
)`}
          </pre>
        </div>
      </CardContent>
    </Card>
  );
}
