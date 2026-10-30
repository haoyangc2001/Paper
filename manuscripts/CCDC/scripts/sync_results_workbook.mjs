// Narrow update of the existing workbook from the checked Python source tables.
// Usage: bundled node sync_results_workbook.mjs <payload.json> <runtime-junction-dir>
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const [payloadPath, runtimeDir] = process.argv.slice(2);
const requireRuntime = createRequire(path.join(runtimeDir, 'loader.mjs'));
const { FileBlob, SpreadsheetFile } = await import(
  pathToFileURL(requireRuntime.resolve('@oai/artifact-tool')).href);
const payload = JSON.parse(await fs.readFile(payloadPath, 'utf8'));
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(payload.workbook));
console.log((await workbook.inspect({kind:'sheet', include:'id,name', maxChars:1800})).ndjson);
await fs.mkdir(payload.review_dir, {recursive:true});
async function preview(name) {
  const blob = await workbook.render({sheetName:'metadata', range:'B14:F18', scale:2, format:'png'});
  await fs.writeFile(path.join(payload.review_dir, name), new Uint8Array(await blob.arrayBuffer()));
}
await preview('workbook-before.png');
if (process.argv.includes('--inspect-only')) process.exit(0);

const changes = [];
for (const [sheetName, rows] of Object.entries(payload.sheets)) {
  const sheet = workbook.worksheets.getItem(sheetName);
  for (let r = 0; r < rows.length; r++) {
    for (let c = 0; c < rows[r].length; c++) {
      const value = rows[r][c];
      const cell = sheet.getCell(r, c);
      const old = cell.values[0]?.[0] ?? '';
      if (old === value || ((old === '' || old == null) && (value === '' || value == null))) continue;
      if (typeof old === 'number' && typeof value === 'number' && Math.abs(old-value)<1e-12) continue;
      changes.push({sheet:sheetName, row:r+1, column:c+1, old, value});
      cell.values = [[value ?? null]];
    }
  }
}
workbook.recalculate();
console.log((await workbook.inspect({kind:'table', range:'metadata!A13:F18',
  include:'values,formulas', tableMaxRows:6, tableMaxCols:6, maxChars:2500})).ndjson);
console.log((await workbook.inspect({kind:'match',
  searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!',
  options:{useRegex:true,maxResults:20}, maxChars:1000})).ndjson);
await preview('workbook-after.png');
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(payload.workbook);
await fs.writeFile(path.join(payload.review_dir,'workbook_changes.json'), JSON.stringify(changes,null,2)+'\n');
console.log(`Updated ${changes.length} cells; retained sheet names/order and unrelated data.`);
