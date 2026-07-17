import { defineTool } from "@lovable.dev/mcp-js";
import { z } from "zod";

export default defineTool({
  name: "validate_ml_results_json",
  title: "Validate ML results JSON",
  description:
    "Validate a multi-ML-classifier results JSON payload against the dashboard's expected schema. Returns whether it is valid plus any errors and warnings by field.",
  inputSchema: {
    json: z
      .string()
      .min(1)
      .describe("Full ML results JSON as a string. Must parse as a JSON object."),
  },
  annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
  handler: async ({ json }) => {
    const { validateMLResultsSchema, formatValidationMessages } = await import(
      "@/utils/jsonSchemaValidator"
    );
    let parsed: unknown;
    try {
      parsed = JSON.parse(json);
    } catch (e) {
      return {
        content: [{ type: "text", text: `Invalid JSON: ${(e as Error).message}` }],
        isError: true,
      };
    }
    const result = validateMLResultsSchema(parsed);
    const text = [
      result.isValid ? "✅ Valid ML results JSON" : "❌ Invalid ML results JSON",
      "",
      ...formatValidationMessages(result),
    ].join("\n");
    return {
      content: [{ type: "text", text }],
      structuredContent: {
        isValid: result.isValid,
        errors: result.errors,
        warnings: result.warnings,
      },
    };
  },
});