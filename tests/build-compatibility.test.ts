import { expect, test } from "bun:test";
import { transform } from "esbuild";
import { resolveConfig } from "vite";

test("the development compiler accepts React state destructuring for Vite browser targets", async () => {
  const config = await resolveConfig({ mode: "development" }, "build");
  const esbuild = config.esbuild;
  if (!esbuild) throw new Error("The compiler compatibility settings are missing");

  expect(esbuild.supported?.destructuring).toBe(true);
  const result = await transform(
    "const [copied, setCopied] = useState(false); const { url } = settings;",
    {
      target: ["chrome87", "edge88", "es2020", "firefox78", "safari14"],
      supported: esbuild.supported,
    },
  );
  expect(result.code).toContain("[copied, setCopied]");
  expect(result.code).toContain("{ url }");
});