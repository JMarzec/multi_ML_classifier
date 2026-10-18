# Project engineering rules

- Both R pipelines must preserve sample identity explicitly and use named, feature-wise preprocessing; this prevents label and expression misalignment.
- Held-out evaluations fit preprocessing and supervised feature selection on training rows only; this prevents optimistic leakage.
- Results validation is shared by uploads and stateless agent tools; this prevents divergent accepted schemas.
- Shared analytics selectors define numeric conversion, missing values and survival cohort placement; this keeps displays and exports consistent.
- Regression tests cover scientific data rules and malformed results, not presentation snapshots; this protects consequential behavior.- Survival forest plotting is isolated from tab data preparation; this keeps rendering changes independent of statistical selectors.
- JavaScript regression tests live outside browser source and use Bun; this avoids introducing test runtime globals into generated clients.
- R download files in public/scripts must match the canonical scripts directory; this ensures website downloads include tested corrections.
- bun.lock is the sole dependency lockfile; this prevents stale package resolutions from competing with patched versions.
- Vite must explicitly preserve supported destructuring syntax in esbuild; this avoids unsupported lowering with the patched compiler while retaining the other browser-target checks.
- Report HTML and embedded SVG must encode uploaded text with the shared escapeHtml helper at insertion, leaving only generated markup unescaped; this preserves scientific labels without allowing script injection.
