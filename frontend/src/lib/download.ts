import type { SimResultBundle } from "@/api/types"

/** Minimal shape of an ECharts instance we need for PNG export. */
export interface EChartsLike {
  getDataURL: (opts?: Record<string, unknown>) => string
}

export function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

export function downloadDataUrl(filename: string, dataUrl: string) {
  const a = document.createElement("a")
  a.href = dataUrl
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
}

export function downloadJson(result: SimResultBundle, filename: string) {
  downloadBlob(filename, new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }))
}

/** Tidy per-day table of the scalar time series (spatial grids / summary /
 * config don't fit a per-day table - those live in the JSON export). */
export function resultToCsv(result: SimResultBundle): string {
  const days = result.days
  const cols: { header: string; values: number[] }[] = [
    { header: "day", values: days.map((d) => d + 1) },
  ]
  for (const p of result.pathogen_names) {
    if (result.u5_prevalence[p]) cols.push({ header: `${p}_u5_prevalence`, values: result.u5_prevalence[p] })
    if (result.all_ages_prevalence[p]) cols.push({ header: `${p}_all_ages_prevalence`, values: result.all_ages_prevalence[p] })
    if (result.cumulative_u5_illness_days[p]) cols.push({ header: `${p}_cumulative_u5_illness_days`, values: result.cumulative_u5_illness_days[p] })
  }
  cols.push({ header: "mean_household_wealth", values: result.mean_household_wealth })
  cols.push({ header: "cumulative_care_seeking_events", values: result.cumulative_care_seeking_events })
  const cum = result.cumulative_care_seeking_events
  cols.push({ header: "daily_care_seeking_events", values: cum.map((v, i) => (i === 0 ? v : v - cum[i - 1])) })
  const routes = result.campy_daily_infections_by_route
  if (routes) for (const r of Object.keys(routes)) cols.push({ header: `campy_${r}_infections`, values: routes[r] })

  const header = cols.map((c) => c.header).join(",")
  const rows = days.map((_, i) => cols.map((c) => c.values[i] ?? "").join(","))
  return [header, ...rows].join("\n")
}

export function downloadCsv(result: SimResultBundle, filename: string) {
  downloadBlob(filename, new Blob([resultToCsv(result)], { type: "text/csv;charset=utf-8" }))
}

export function downloadChartPng(chart: EChartsLike | undefined, filename: string) {
  if (!chart) return
  downloadDataUrl(filename, chart.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: "#ffffff" }))
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = "anonymous"
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

/** The spatial map is an ECharts canvas layered over a plain <img> basemap, so
 * a bare getDataURL misses the satellite layer. Composite basemap + overlay
 * (same-origin img, no CORS taint) onto one canvas before download. */
export async function downloadSpatialPng(chart: EChartsLike | undefined, basemapSrc: string, filename: string) {
  if (!chart) return
  const overlayUrl = chart.getDataURL({ type: "png", pixelRatio: 2 }) // transparent bg keeps the basemap visible
  const [overlay, base] = await Promise.all([loadImage(overlayUrl), loadImage(basemapSrc)])
  const w = overlay.width
  const h = overlay.height
  const canvas = document.createElement("canvas")
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  ctx.drawImage(base, 0, 0, w, h)
  ctx.drawImage(overlay, 0, 0, w, h)
  canvas.toBlob((blob) => {
    if (blob) downloadBlob(filename, blob)
  }, "image/png")
}
