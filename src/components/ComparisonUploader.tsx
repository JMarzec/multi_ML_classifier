import { useState, useCallback } from "react";
import { Upload, FileJson, X, Plus, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { MLResults } from "@/types/ml-results";

import { parseMLResults } from "@/utils/jsonSchemaValidator";

const MAX_FILES = 4;

const RUN_LABELS = ["Run A", "Run B", "Run C", "Run D"];
const RUN_COLORS = [
  { bg: "bg-primary/10", border: "border-primary/30", text: "text-primary", badge: "bg-primary/20" },
  { bg: "bg-secondary/10", border: "border-secondary/30", text: "text-secondary", badge: "bg-secondary/20" },
  { bg: "bg-accent/10", border: "border-accent/30", text: "text-accent", badge: "bg-accent/20" },
  { bg: "bg-warning/10", border: "border-warning/30", text: "text-warning", badge: "bg-warning/20" },
];

interface ComparisonUploaderProps {
  onFilesLoaded: (files: { name: string; data: MLResults }[]) => void;
  currentFiles: { name: string; data: MLResults }[];
}

export function ComparisonUploader({ onFilesLoaded, currentFiles }: ComparisonUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);

      if (!file.name.toLowerCase().endsWith(".json")) {
        setError("Please upload a JSON file");
        return;
      }

      if (currentFiles.some((f) => f.name === file.name)) {
        setError("A file with this name is already loaded");
        return;
      }

      if (currentFiles.length >= MAX_FILES) {
        setError(`Maximum ${MAX_FILES} files for comparison`);
        return;
      }

      setIsLoading(true);
      try {
        const text = await file.text();
        const data = parseMLResults(text);


        onFilesLoaded([...currentFiles, { name: file.name, data }]);
      } catch (error) {
        setError(error instanceof Error ? error.message : "Unable to load results file");
      } finally {
        setIsLoading(false);
      }
    },
    [currentFiles, onFilesLoaded]
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
      e.target.value = "";
    },
    [handleFile]
  );

  const removeFile = (name: string) => {
    onFilesLoaded(currentFiles.filter((f) => f.name !== name));
  };

  const getUploadMessage = () => {
    if (isLoading) return "Parsing file…";
    if (currentFiles.length === 0) return "Upload first analysis file";
    if (currentFiles.length === 1) return "Add second file for comparison";
    return `Add file ${currentFiles.length + 1} (optional, max ${MAX_FILES})`;
  };

  return (
    <div className="space-y-4">
      {/* Current files */}
      {currentFiles.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {currentFiles.map((file, index) => {
            const colors = RUN_COLORS[index];
            return (
              <div
                key={file.name}
                className={cn(
                  "flex items-center gap-2 px-3 py-2 rounded-lg border",
                  colors.bg,
                  colors.border
                )}
              >
                <FileJson className={cn("w-4 h-4", colors.text)} />
                <span className="text-sm font-medium truncate max-w-[200px]">{file.name}</span>
                <span
                  className={cn(
                    "text-xs px-2 py-0.5 rounded-full",
                    colors.badge,
                    colors.text
                  )}
                >
                  {RUN_LABELS[index]}
                </span>
                <button
                  onClick={() => removeFile(file.name)}
                  className="p-1 hover:bg-muted rounded-full transition-colors"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            );
          })}
        </div>
      )}

      {/* Upload area */}
      {currentFiles.length < MAX_FILES && (
        <div
          className={cn(
            "relative border-2 border-dashed rounded-xl p-6 transition-all duration-300",
            "bg-card/50 backdrop-blur-sm cursor-pointer",
            isDragging && "border-primary bg-primary/5",
            !isDragging && "border-border hover:border-primary/50"
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

          <div className="flex items-center justify-center gap-3 pointer-events-none">
            <div className="p-2 rounded-full bg-primary/20">
              {isLoading ? (
                <Loader2 className="w-5 h-5 text-primary animate-spin" />
              ) : currentFiles.length === 0 ? (
                <Upload className="w-5 h-5 text-primary" />
              ) : (
                <Plus className="w-5 h-5 text-primary" />
              )}
            </div>
            <div className="text-center">
              <p className="text-sm font-medium">{getUploadMessage()}</p>
              <p className="text-xs text-muted-foreground">
                {isLoading ? "Please wait" : "Drop JSON or click to browse"}
              </p>
            </div>
          </div>
        </div>
      )}

      {error && <p className="text-sm text-destructive">{error}</p>}

      {currentFiles.length >= 2 && (
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => onFilesLoaded([])} className="flex-1">
            Clear all files
          </Button>
          {currentFiles.length < MAX_FILES && (
            <p className="text-xs text-muted-foreground self-center">
              {MAX_FILES - currentFiles.length} more file(s) can be added
            </p>
          )}
        </div>
      )}
    </div>
  );
}
