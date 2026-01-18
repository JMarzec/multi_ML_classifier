import { useState, useCallback } from "react";
import { Upload, FileJson, X, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { MLResults } from "@/types/ml-results";

interface ComparisonUploaderProps {
  onFilesLoaded: (files: { name: string; data: MLResults }[]) => void;
  currentFiles: { name: string; data: MLResults }[];
}

export function ComparisonUploader({ onFilesLoaded, currentFiles }: ComparisonUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(async (file: File) => {
    setError(null);

    if (!file.name.endsWith('.json')) {
      setError("Please upload a JSON file");
      return;
    }

    if (currentFiles.some(f => f.name === file.name)) {
      setError("A file with this name is already loaded");
      return;
    }

    if (currentFiles.length >= 2) {
      setError("Maximum 2 files for comparison");
      return;
    }

    try {
      const text = await file.text();
      const data = JSON.parse(text) as MLResults;

      if (!data.model_performance || !data.metadata) {
        setError("Invalid file format");
        return;
      }

      onFilesLoaded([...currentFiles, { name: file.name, data }]);
    } catch {
      setError("Failed to parse JSON file");
    }
  }, [currentFiles, onFilesLoaded]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = '';
  }, [handleFile]);

  const removeFile = (name: string) => {
    onFilesLoaded(currentFiles.filter(f => f.name !== name));
  };

  return (
    <div className="space-y-4">
      {/* Current files */}
      {currentFiles.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {currentFiles.map((file, index) => (
            <div
              key={file.name}
              className={cn(
                "flex items-center gap-2 px-3 py-2 rounded-lg border",
                index === 0 ? "bg-primary/10 border-primary/30" : "bg-secondary/10 border-secondary/30"
              )}
            >
              <FileJson className={cn("w-4 h-4", index === 0 ? "text-primary" : "text-secondary")} />
              <span className="text-sm font-medium truncate max-w-[200px]">{file.name}</span>
              <span className={cn(
                "text-xs px-2 py-0.5 rounded-full",
                index === 0 ? "bg-primary/20 text-primary" : "bg-secondary/20 text-secondary"
              )}>
                {index === 0 ? "Run A" : "Run B"}
              </span>
              <button
                onClick={() => removeFile(file.name)}
                className="p-1 hover:bg-muted rounded-full transition-colors"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Upload area */}
      {currentFiles.length < 2 && (
        <div
          className={cn(
            "relative border-2 border-dashed rounded-xl p-6 transition-all duration-300",
            "bg-card/50 backdrop-blur-sm cursor-pointer",
            isDragging && "border-primary bg-primary/5",
            !isDragging && "border-border hover:border-primary/50"
          )}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept=".json"
            onChange={handleChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
          />
          
          <div className="flex items-center justify-center gap-3 pointer-events-none">
            <div className="p-2 rounded-full bg-primary/20">
              {currentFiles.length === 0 ? (
                <Upload className="w-5 h-5 text-primary" />
              ) : (
                <Plus className="w-5 h-5 text-primary" />
              )}
            </div>
            <div className="text-center">
              <p className="text-sm font-medium">
                {currentFiles.length === 0
                  ? "Upload first analysis file"
                  : "Add second file for comparison"}
              </p>
              <p className="text-xs text-muted-foreground">
                Drop JSON or click to browse
              </p>
            </div>
          </div>
        </div>
      )}

      {error && (
        <p className="text-sm text-destructive">{error}</p>
      )}

      {currentFiles.length === 2 && (
        <Button
          variant="outline"
          size="sm"
          onClick={() => onFilesLoaded([])}
          className="w-full"
        >
          Clear and start new comparison
        </Button>
      )}
    </div>
  );
}
