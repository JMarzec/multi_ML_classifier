import { describe, expect, test } from "bun:test";
import summarize from "../src/lib/mcp/tools/summarize-run";
import compare from "../src/lib/mcp/tools/compare-runs";
import validate from "../src/lib/mcp/tools/validate-json";
const run = JSON.stringify({metadata:{config:{target_variable:"status"}},model_performance:{rf:{auroc:{mean:0},accuracy:{mean:.75}}}});
describe("stateless result tools", () => {
 test("malformed payloads fail instead of producing success summaries", async () => {
  for (const json of ["null", "42", "{}", '{"model_performance":{"rf":null}}']) {
   expect((await summarize.handler({json}, {} as never)).isError).toBe(true);
   expect((await compare.handler({json_a:json,json_b:run}, {} as never)).isError).toBe(true);
  }
 });
 test("zero AUROC is retained in tool summaries", async () => {
  const result = await summarize.handler({json:run}, {} as never);
  expect(result.structuredContent?.models).toEqual([{model:"rf",auroc:0,accuracy:.75,f1:null}]);
 });
 test("validated JSON is accepted through the same entry point", async () => {
  expect((await validate.handler({json:run}, {} as never)).structuredContent?.isValid).toBe(true);
 });
 test("missing F1 is not compared as zero", async () => {
  const result = await compare.handler({json_a:run,json_b:run}, {} as never);
  expect(result.structuredContent?.models).toBeDefined();
  expect(JSON.stringify(result.structuredContent)).toContain('"f1_delta":null');
 });
});
