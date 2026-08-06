// Formats a slider/number value for display: integers show with no
// decimals, everything else shows up to 4 significant digits with
// trailing zeros trimmed (so 0.3005 reads as "0.3005" and 5 reads as "5",
// not "5.0000") - most registry fields span very different magnitudes
// (e.g. 0.0001 to 10000), so a single fixed decimal-place count would
// either truncate the small ones or clutter the large ones.
export function formatNumber(value: number, isInteger: boolean): string {
  if (isInteger) return String(Math.round(value))
  if (value === 0) return "0"
  return Number(value.toPrecision(4)).toString()
}

// Sensible slider step when the registry doesn't specify one (ui_step is
// only set for a handful of whole-number fields - see parameter_registry.py's
// ParamMeta docstring). 1000 discrete positions across the range is fine
// granularity for both very narrow (e.g. 0-0.004) and very wide (e.g.
// 0-10000) fields without needing per-field tuning.
export function defaultStep(uiMin: number, uiMax: number): number {
  const span = uiMax - uiMin
  if (span <= 0) return 1
  return span / 1000
}
