#!/usr/bin/env node
/**
 * Bundle tui-replay traces into a single self-contained HTML file.
 *
 * Run from the repo root after e2e tests complete:
 *   node e2e/scripts/bundle-replay-html.mjs
 *
 * Input:  e2e/tui-traces/  (default @microsoft/tui-test output dir)
 * Output: tui-replay-viewer/replay.html  (uploaded as a GHA artifact)
 */
import { createReplayDataSource } from 'tui-replay';
import { readFile, writeFile, mkdir, access } from 'node:fs/promises';
import { resolve, join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '../..');

// tui-replay/dist/ — resolved via ESM so package exports are honoured
const tuiReplayDist = dirname(fileURLToPath(import.meta.resolve('tui-replay')));

const tracesDir = resolve(repoRoot, 'e2e/tui-traces');
const outputDir = resolve(repoRoot, 'tui-replay-viewer');
const outputFile = join(outputDir, 'replay.html');

// ── exact strings to patch in client.js ────────────────────────────────────
const SELECTORS_IMPORT =
  'import { annotationsForFrame, frameIndexAtTime, timelineItems } from "../preview/selectors.js";';

// Lines 166-172 of dist/viewer/client.js (0.4.x)
const FETCH_ORIGINAL = `async function fetchPreviewModel() {
    const response = await fetch("/api/traces");
    if (!response.ok) {
        throw new Error(\`Unable to load traces: \${response.status}\`);
    }
    return (await response.json());
}`;
const FETCH_PATCHED = `async function fetchPreviewModel() {
    return __INLINE_MODEL__;
}`;

// Lines 140-149 of dist/viewer/client.js (0.4.x)
const CONNECT_ORIGINAL = `function connectLiveUpdates() {
    if (!("EventSource" in window)) {
        startPollingLiveUpdates();
        return;
    }
    const events = new EventSource("/api/events");
    events.addEventListener("model", (event) => {
        applyModelUpdate(JSON.parse(event.data));
    });
}`;
const CONNECT_PATCHED = `function connectLiveUpdates() {
    /* static mode: no live updates */
}`;
// ───────────────────────────────────────────────────────────────────────────

async function main() {
  // Gracefully skip when traces haven't been written yet (e.g. tests skipped)
  try {
    await access(tracesDir);
  } catch {
    console.log(`tui-traces dir not found at ${tracesDir} — skipping HTML bundle.`);
    process.exit(0);
  }

  console.log(`Loading traces from ${tracesDir} …`);
  const dataSource = createReplayDataSource({
    inputs: [tracesDir],
    projectRoot: repoRoot,
  });
  const model = await dataSource.load();

  if (model.traces.length === 0) {
    console.log('No traces found — skipping HTML bundle.');
    process.exit(0);
  }
  console.log(`Found ${model.traces.length} trace(s).`);

  // ── Load tui-replay dist assets ──────────────────────────────────────────
  // renderIndexHtml is internal (not in the public index.js export) so we
  // import it directly from the dist path.
  const { renderIndexHtml } = await import(
    pathToFileURL(join(tuiReplayDist, 'server/html.js')).href
  );

  const [rawClientJs, rawSelectorsJs] = await Promise.all([
    readFile(join(tuiReplayDist, 'viewer/client.js'), 'utf8'),
    readFile(join(tuiReplayDist, 'preview/selectors.js'), 'utf8'),
  ]);

  // ── Patch client.js for static/embedded use ──────────────────────────────
  let clientJs = rawClientJs;

  // 1. Remove the ES module import (selectors will be inlined above it)
  if (!clientJs.includes(SELECTORS_IMPORT)) {
    throw new Error(
      'Could not find selectors import in client.js — tui-replay may have updated. ' +
      'Please update the SELECTORS_IMPORT constant in bundle-replay-html.mjs.'
    );
  }
  clientJs = clientJs.replace(SELECTORS_IMPORT + '\n', '');

  // 2. Replace the live fetch with a return of the inlined model
  if (!clientJs.includes(FETCH_ORIGINAL)) {
    throw new Error(
      'Could not find fetchPreviewModel body in client.js — tui-replay may have updated. ' +
      'Please update FETCH_ORIGINAL in bundle-replay-html.mjs.'
    );
  }
  clientJs = clientJs.replace(FETCH_ORIGINAL, FETCH_PATCHED);

  // 3. Disable live-reload SSE/polling (no server in static mode)
  if (!clientJs.includes(CONNECT_ORIGINAL)) {
    throw new Error(
      'Could not find connectLiveUpdates body in client.js — tui-replay may have updated. ' +
      'Please update CONNECT_ORIGINAL in bundle-replay-html.mjs.'
    );
  }
  clientJs = clientJs.replace(CONNECT_ORIGINAL, CONNECT_PATCHED);

  // Strip sourcemap comment (optional — keeps file clean in artifact viewer)
  clientJs = clientJs.replace(/\n\/\/#\s*sourceMappingURL=client\.js\.map\s*$/, '');

  // ── Prepare selectors for inline use ─────────────────────────────────────
  // Remove `export` keyword so the functions are available in the same
  // module scope as client.js (they're no longer imported — they're just
  // declared above client.js in the same <script type="module"> block).
  const selectorsInline = rawSelectorsJs
    .replace(/^export function /gm, 'function ')
    .replace(/\n\/\/#\s*sourceMappingURL=selectors\.js\.map\s*$/, '');

  // ── Embed model JSON ──────────────────────────────────────────────────────
  // JSON.stringify is safe inside a JS string but escape </script> sequences
  // just in case trace content contains them.
  const modelJsonString = JSON.stringify(model).replace(/<\/script>/gi, '<\\/script>');

  // ── Assemble HTML ─────────────────────────────────────────────────────────
  const htmlTemplate = renderIndexHtml();

  const SCRIPT_TAG = '<script type="module" src="/assets/client.js"></script>';
  if (!htmlTemplate.includes(SCRIPT_TAG)) {
    throw new Error(
      'Could not find the client script tag in the HTML template — ' +
      'tui-replay may have updated. Please update SCRIPT_TAG in bundle-replay-html.mjs.'
    );
  }

  const inlinedHtml = htmlTemplate.replace(
    SCRIPT_TAG,
    `<script type="module">
/* tui-replay selectors (inlined) */
${selectorsInline}

/* trace model (embedded at bundle time) */
const __INLINE_MODEL__ = JSON.parse(${JSON.stringify(modelJsonString)});

/* tui-replay client (patched for static mode) */
${clientJs}
</script>`
  );

  // ── Write output ──────────────────────────────────────────────────────────
  await mkdir(outputDir, { recursive: true });
  await writeFile(outputFile, inlinedHtml, 'utf8');

  const sizeKb = (Buffer.byteLength(inlinedHtml, 'utf8') / 1024).toFixed(1);
  console.log(`✓ Wrote ${outputFile} (${sizeKb} KB, ${model.traces.length} trace(s))`);
}

main().catch((err) => {
  console.error('bundle-replay-html failed:', err.message ?? err);
  process.exit(1);
});
