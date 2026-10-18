import { describe, expect, test } from "bun:test";
import { confusionMetrics, normalizePerGene, normalizeModelRiskScores, patientRiskCohort, toFiniteNumber } from "../src/utils/analytics";
import { parseMLResults, validateMLResultsSchema } from "../src/utils/jsonSchemaValidator";
import type { MLResults, ProfileRanking } from "@/types/ml-results";

const run = {metadata: {config: {target_variable: "status"}}, model_performance: {rf: {accuracy: {mean: .75}}}};
describe("results ingestion", () => {
  test("rejects null models without throwing from validation", () => {
    expect(validateMLResultsSchema({...run, model_performance: {rf: null}}).isValid).toBe(false);
  });
  test("rejects malformed ranking collections", () => {
    expect(() => parseMLResults(JSON.stringify({...run, profile_ranking: {all_rankings: {bad: true}}}))).toThrow("profile_ranking.all_rankings");
  });
  test("rejects non-object roots and empty runs", () => {
    for (const input of ["null", "42", "[]", "{}"] ) expect(() => parseMLResults(input)).toThrow();
  });
  test("retains measured zero and supplies absent optional arrays", () => {
    const result = parseMLResults(JSON.stringify({...run, model_performance: {rf: {accuracy: {mean: 0}}}}));
    expect(result.model_performance.rf?.accuracy?.mean).toBe(0);
    expect(result.selected_features).toEqual([]);
  });
});
describe("scientific selectors", () => {
  test("legacy NA metric summaries become unavailable rather than crashing uploads", () => {
    const result = parseMLResults(JSON.stringify({...run,model_performance:{rf:{accuracy:{mean:.7},auroc:{mean:"NA"}}}}));
    expect(result.model_performance.rf?.auroc).toBeUndefined();
  });
  test("positive-class counts agree with sensitivity and precision", () => {
    const metrics = confusionMetrics({tn: 8, fp: 2, fn: 1, tp: 1});
    expect(metrics.recall).toBe(.5);
    expect(metrics.precision).toBe(1 / 3);
    expect(metrics.specificity).toBe(.8);
  });
  test("undefined rates are unavailable, not zero", () => {
    expect(confusionMetrics({tp: 0, tn: 0, fp: 0, fn: 0}).accuracy).toBeUndefined();
  });
  test("numeric survival strings normalize identically from keyed objects", () => {
    expect(normalizePerGene({GENE1: {cox_p: "0.001", cox_hr: "2"}})[0].cox_p).toBe(.001);
    expect(normalizeModelRiskScores({soft_vote: {stats: {cox_hr: "2"}, cutoff: "0.3"}})[0].cutoff).toBe(.3);
    expect(toFiniteNumber("")).toBeUndefined();
  });
  test("patient zero probability never becomes confidence and uses exported cutoff", () => {
    const data = {...run, survival_analysis: {model_risk_scores: [{model: "soft_vote", cutoff: .3, high_group_rule: "probability > cutoff"}]}} as unknown as MLResults;
    expect(patientRiskCohort(data, {ensemble_probability: 0, confidence: .9} as ProfileRanking)).toBe("low");
    expect(patientRiskCohort(data, {ensemble_probability: .4} as ProfileRanking)).toBe("high");
    expect(patientRiskCohort(data, {ensemble_probability: .3} as ProfileRanking)).toBe("low");
  });
  test("legacy runs without cohort cutoffs do not invent patient placement", () => {
    expect(patientRiskCohort(run as unknown as MLResults, {ensemble_probability: .8} as ProfileRanking)).toBeUndefined();
  });
});