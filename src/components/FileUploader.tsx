import { useState, useCallback } from "react";
import { Upload, FileJson, AlertCircle, CheckCircle2, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { MLResults } from "@/types/ml-results";

interface FileUploaderProps {
  onDataLoaded: (data: MLResults) => void;
}

export function FileUploader({ onDataLoaded }: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      setSuccess(false);

      if (!file.name.toLowerCase().endsWith(".json")) {
        setError("Please upload a JSON file");
        return;
      }

      setIsLoading(true);
      try {
        const text = await file.text();
        const data = JSON.parse(text) as MLResults;

        // Validate structure
        if (!data.model_performance || !data.metadata) {
          setError("Invalid file format. Please upload results from the R script.");
          return;
        }

        setSuccess(true);
        setTimeout(() => {
          onDataLoaded(data);
        }, 200);
      } catch {
        setError("Failed to parse JSON file");
      } finally {
        setIsLoading(false);
      }
    },
    [onDataLoaded]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const file = e.dataTransfer.files?.[0];
      if (file) void handleFile(file);
    },
    [handleFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) void handleFile(file);
      // Allow re-uploading the same file name repeatedly
      e.target.value = "";
    },
    [handleFile]
  );

  return (
    <div
      className={cn(
        "relative border-2 border-dashed rounded-xl p-12 transition-all duration-300",
        "bg-card/50 backdrop-blur-sm",
        isDragging && "border-primary bg-primary/5 scale-[1.02]",
        !isDragging && "border-border hover:border-primary/50",
        success && "border-accent bg-accent/5",
        error && "border-destructive bg-destructive/5"
      )}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      <input
        type="file"
        accept=".json,application/json"
        onChange={handleChange}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
      />

      <div className="flex flex-col items-center gap-4 pointer-events-none">
        <div
          className={cn(
            "p-4 rounded-full transition-colors",
            success
              ? "bg-accent/20"
              : error
                ? "bg-destructive/20"
                : "bg-primary/20"
          )}
        >
          {isLoading ? (
            <Loader2 className="w-10 h-10 text-primary animate-spin" />
          ) : success ? (
            <CheckCircle2 className="w-10 h-10 text-accent" />
          ) : error ? (
            <AlertCircle className="w-10 h-10 text-destructive" />
          ) : isDragging ? (
            <FileJson className="w-10 h-10 text-primary animate-pulse" />
          ) : (
            <Upload className="w-10 h-10 text-primary" />
          )}
        </div>

        <div className="text-center">
          <p className="text-lg font-medium text-foreground">
            {isLoading
              ? "Parsing file…"
              : success
                ? "File loaded successfully!"
                : error
                  ? error
                  : isDragging
                    ? "Drop your file here"
                    : "Drag & drop your results JSON"}
          </p>
          <p className="text-sm text-muted-foreground mt-1">
            {!error && !success && !isLoading && "or click to browse files"}
          </p>
        </div>
      </div>
    </div>
  );
}
