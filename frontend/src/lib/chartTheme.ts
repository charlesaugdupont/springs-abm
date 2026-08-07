// Chart color roles, sourced from the dataviz skill's validated reference
// palette (references/palette.md) - not shadcn's default --chart-* tokens,
// which the "Nova" preset ships as pure zero-chroma grays, unusable for
// distinguishing series. Validated via scripts/validate_palette.js for
// both this app's two series (rota=slot1 blue, campy=slot2 orange): all
// checks pass in both light and dark mode (worst adjacent CVD ΔE 24.7
// light / 26.8 dark, well clear of the >=8 target).

export function isDarkMode(): boolean {
  return document.documentElement.classList.contains("dark")
}

interface ThemeColors {
  rota: string
  campy: string
  primary: string
  secondary: string
  muted: string
  gridline: string
  baseline: string
  surface: string
}

const LIGHT: ThemeColors = {
  rota: "#2a78d6",
  campy: "#eb6834",
  primary: "#0b0b0b",
  secondary: "#52514e",
  muted: "#898781",
  gridline: "#e1e0d9",
  baseline: "#c3c2b7",
  surface: "#fcfcfb",
}

const DARK: ThemeColors = {
  rota: "#3987e5",
  campy: "#d95926",
  primary: "#ffffff",
  secondary: "#c3c2b7",
  muted: "#898781",
  gridline: "#2c2c2a",
  baseline: "#383835",
  surface: "#1a1a19",
}

export function chartColors(): ThemeColors {
  return isDarkMode() ? DARK : LIGHT
}

export const PATHOGEN_COLOR: Record<string, keyof Pick<ThemeColors, "rota" | "campy">> = {
  rota: "rota",
  campy: "campy",
}

// Campylobacter's three transmission routes. Display/stacking order matters:
// orange (campy's family hue), aqua, yellow - kept in this order so orange and
// yellow never sit adjacent (aqua between them), the one pair that fails the
// all-pairs floor. Validated via scripts/validate_palette.js in both modes
// (worst adjacent CVD ΔE 9.1 light / 8.4 dark). Light-mode aqua/yellow are
// sub-3:1 on the surface, so CampyRouteAreaChart adds direct end-labels (relief).
export const CAMPY_ROUTE_ORDER = ["zoonotic", "fecal_oral", "food_borne"] as const

export const CAMPY_ROUTE_LABEL: Record<string, string> = {
  zoonotic: "Zoonotic",
  fecal_oral: "Fecal-oral",
  food_borne: "Food-borne",
}

export function campyRouteColors(): Record<string, string> {
  // Light-mode teal/gold are deepened (#12946a / #bd7d00, vs the brighter
  // #1baf7a / #eda100) so all three routes clear >=3:1 contrast on the light
  // surface - the chart no longer carries direct end-labels, so it can't lean
  // on them for the sub-3:1 relief. Validated via the dataviz validate_palette
  // script; the residual orange<->teal CVD is covered by the 2px surface-gap
  // separators between the stacked bands. Dark mode already passes as-is.
  return isDarkMode()
    ? { zoonotic: "#d95926", fecal_oral: "#199e70", food_borne: "#c98500" }
    : { zoonotic: "#eb6834", fecal_oral: "#12946a", food_borne: "#bd7d00" }
}

// Sequential single-hue (blue) ramp, light->dark, steps 100-700 - used for
// the spatial heatmap's magnitude encoding (never a rainbow "Hot" scale).
export const SEQUENTIAL_BLUE = [
  "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
  "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

// Root-level ECharts option fragment shared by every chart: just the grid
// box + animation flag. `containLabel: true` lets the grid auto-reserve room
// for tick labels AND axis names (ECharts >=5.4 / 6.x), so axis names no
// longer clip against the card edge the way a fixed `grid.left` did.
export function baseGridAxisOption() {
  return {
    // Entrance animation ("lines grow in") is off by default across every
    // chart: a scientific tool showing a completed simulation's results
    // should render its final state immediately, not mid-sweep - and a
    // screenshot taken right after setOption() would otherwise capture
    // that transient partially-drawn state instead of the real data.
    animation: false,
    // bottom is generous (not the ~14 that containLabel alone reserves) so the
    // x-axis `name` ("Day") drawn at nameGap below the tick labels doesn't clip
    // against the canvas bottom edge.
    grid: { left: 12, right: 20, top: 30, bottom: 40, containLabel: true },
  }
}

// Per-axis style fragment matching the skill's mark specs: hairline (1px)
// solid recessive gridlines, axis line + text in muted ink. This MUST be
// spread inside each `xAxis`/`yAxis` object - ECharts ignores axisLine /
// axisLabel / splitLine at the option root (the previous bug: they silently
// no-op'd, so ticks/gridlines fell back to defaults and could vanish in
// cramped cards). When an axis also needs a tick formatter, merge it:
// `axisLabel: { ...axisStyle(colors).axisLabel, formatter }`.
export function axisStyle(colors: ThemeColors) {
  return {
    axisLine: { lineStyle: { color: colors.baseline } },
    axisLabel: { color: colors.muted, fontSize: 11 },
    splitLine: { lineStyle: { color: colors.gridline, width: 1, type: "solid" as const } },
  }
}

// Shared axis-trigger tooltip formatter that prefixes the day ("Day 132")
// on the header row - ECharts' `valueFormatter` only formats the value rows,
// not the axis header, so the bare day number showed otherwise. `valueFmt`
// formats each series value (e.g. as a percentage). Typed loosely because the
// ECharts callback-params type is a broad union.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function dayAxisTooltip(valueFmt?: (v: number) => string) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (params: any): string => {
    const arr = Array.isArray(params) ? params : [params]
    if (arr.length === 0) return ""
    const rows = arr
      .map((p) => {
        const raw = Array.isArray(p.value) ? p.value[p.value.length - 1] : p.value
        const val = valueFmt ? valueFmt(raw as number) : `${raw}`
        return `${p.marker ?? ""}${p.seriesName ?? ""}: <b>${val}</b>`
      })
      .join("<br/>")
    return `<span style="font-size:11px;opacity:0.7">Day ${arr[0].axisValue}</span><br/>${rows}`
  }
}
