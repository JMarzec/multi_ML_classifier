import { expect, test } from "bun:test";
import { escapeHtml } from "../src/utils/escapeHtml";
import { buildComparisonReportHTML } from "../src/utils/comparisonReport";
import { buildFeatureImportanceSVG, buildConfusionMatrixSVG } from "../src/utils/chartToSvg";
import { demoData } from "../src/data/demoData";

test("report text encoding preserves labels but neutralizes markup and quoted attributes", () => {
  expect(escapeHtml(`<script>"A&B" 'case'</script>`)).toBe("&lt;script&gt;&quot;A&amp;B&quot; &#39;case&#39;&lt;/script&gt;");
  expect(escapeHtml("BRCA1 / 0 (Neg)" )).toBe("BRCA1 / 0 (Neg)");
  expect(escapeHtml(0)).toBe("0");
});

test("comparison filenames and overlapping features are encoded", () => {
  const payload = '<img src=x onerror="window.injected=true">';
  const data = { ...demoData, selected_features: [payload] };
  const html = buildComparisonReportHTML([{ name: payload, data }, { name: "Control", data }]);
  expect(html).not.toContain(payload);
  expect(html).toContain(escapeHtml(payload));
  expect(html).toContain('<svg xmlns="http://www.w3.org/2000/svg"');
});

test("SVG labels and titles cannot escape text nodes", () => {
  const payload = '</text><script>window.injected=true</script><text>';
  const featureSvg = buildFeatureImportanceSVG([{ feature: "<svg/onload=x>", importance: 1 }], 10, payload);
  expect(featureSvg).toContain(escapeHtml("<svg/onload=x>"));
  expect(featureSvg).toContain(escapeHtml(payload));
  expect(featureSvg).not.toContain("<script>");
  expect(buildConfusionMatrixSVG(1, 1, 0, 0, payload)).toContain(escapeHtml(payload));
});