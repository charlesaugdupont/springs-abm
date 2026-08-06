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

// Sequential single-hue (blue) ramp, light->dark, steps 100-700 - used for
// the spatial heatmap's magnitude encoding (never a rainbow "Hot" scale).
export const SEQUENTIAL_BLUE = [
  "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
  "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

// Shared ECharts option fragments matching the skill's mark specs: hairline
// (1px) solid recessive gridlines, axis text in muted ink, 2px lines
// (applied per-series by each chart component).
export function baseGridAxisOption(colors: ThemeColors) {
  return {
    // Entrance animation ("lines grow in") is off by default across every
    // chart: a scientific tool showing a completed simulation's results
    // should render its final state immediately, not mid-sweep - and a
    // screenshot taken right after setOption() would otherwise capture
    // that transient partially-drawn state instead of the real data.
    animation: false,
    grid: { left: 48, right: 16, top: 24, bottom: 32, containLabel: true },
    axisLine: { lineStyle: { color: colors.baseline } },
    axisLabel: { color: colors.muted, fontSize: 11 },
    splitLine: { lineStyle: { color: colors.gridline, width: 1, type: "solid" as const } },
  }
}
