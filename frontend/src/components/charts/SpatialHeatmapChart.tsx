import { useEffect, useMemo, useState } from "react"
import type { EChartsOption } from "echarts"
import { useECharts } from "@/hooks/useECharts"
import { SEQUENTIAL_BLUE } from "@/lib/chartTheme"
import { spatialAxes, type BasemapMeta } from "@/lib/spatialAxes"
import { Slider } from "@/components/ui/slider"
import { cn } from "@/lib/utils"
import type { EChartsLike } from "@/lib/download"

interface SpatialHeatmapChartProps {
  basemap: BasemapMeta
  spatialDailyGrids: number[][][]
  gridSize: number
  staticLayers?: Record<string, number[][]>
  onReady?: (chart: EChartsLike) => void
}

// Single-hue ramps for the reference density layers (kept distinct from the
// infection blue). POI markers use a single swatch colour each.
const GREY_RAMP = ["#d4d4d8", "#a1a1aa", "#52525b"]
const AMBER_RAMP = ["#fde3a7", "#f0a935", "#b5701a"]
const RIVER_RAMP = ["#bcd8f5", "#5b9bd8"]

type LayerKind = "infection" | "heatmap" | "poi"
interface LayerDef {
  key: string
  label: string
  swatch: string
  kind: LayerKind
  source?: string // key in staticLayers (absent for infection)
  ramp?: string[]
  opacity?: [number, number]
  symbol?: string
}

// Z-order for drawing (later = on top): context/densities below, infection,
// then POI markers on top.
const Z_ORDER = ["rivers", "households", "animals", "infections", "schools", "water", "worship"]

/** Cumulative infection-density heatmap over the Akuse basemap, with a day
 * scrubber and optional static reference overlays (households, animal density,
 * schools/water/worship points, water bodies) the user can toggle on. All
 * layers share the same 25x25 [y][x] frame, so they align with the basemap. */
export function SpatialHeatmapChart({ basemap, spatialDailyGrids, gridSize, staticLayers, onReady }: SpatialHeatmapChartProps) {
  const { containerRef, chartRef } = useECharts()
  const lastDay = spatialDailyGrids.length - 1
  const [day, setDay] = useState(lastDay) // default to the final day - the fullest picture
  const [active, setActive] = useState<Set<string>>(() => new Set(["infections"]))

  const axes = useMemo(() => spatialAxes(basemap, gridSize), [basemap, gridSize])
  const aspectRatio = (basemap.maxx - basemap.minx) / (basemap.maxy - basemap.miny)
  const sl = staticLayers ?? {}

  const zMax = useMemo(() => {
    let max = 0
    for (const grid of spatialDailyGrids) for (const row of grid) for (const v of row) if (v > max) max = v
    return max > 0 ? max : 1
  }, [spatialDailyGrids])

  // Which layers are available depends on what the backend sent.
  const layers = useMemo<LayerDef[]>(() => {
    const defs: LayerDef[] = [
      { key: "infections", label: "Infections", swatch: "#256abf", kind: "infection", ramp: SEQUENTIAL_BLUE, opacity: [0.25, 0.82] },
    ]
    if (sl.household_density) defs.push({ key: "households", label: "Households", swatch: "#52525b", kind: "heatmap", source: "household_density", ramp: GREY_RAMP, opacity: [0.2, 0.6] })
    if (sl.animal_density) defs.push({ key: "animals", label: "Animals", swatch: "#b5701a", kind: "heatmap", source: "animal_density", ramp: AMBER_RAMP, opacity: [0.12, 0.5] })
    if (sl.school) defs.push({ key: "schools", label: "Schools", swatch: "#6d5ac0", kind: "poi", source: "school", symbol: "rect" })
    if (sl.water) defs.push({ key: "water", label: "Water points", swatch: "#2f80c4", kind: "poi", source: "water", symbol: "circle" })
    if (sl.place_of_worship) defs.push({ key: "worship", label: "Worship", swatch: "#b8763a", kind: "poi", source: "place_of_worship", symbol: "diamond" })
    if (sl.natural_water) defs.push({ key: "rivers", label: "Rivers", swatch: "#5b9bd8", kind: "heatmap", source: "natural_water", ramp: RIVER_RAMP, opacity: [0.3, 0.55] })
    return defs
  }, [sl])
  const layerByKey = useMemo(() => Object.fromEntries(layers.map((d) => [d.key, d])), [layers])

  const option = useMemo<EChartsOption>(() => {
    /* eslint-disable @typescript-eslint/no-explicit-any */
    const series: any[] = []
    const visualMap: any[] = []
    const seriesLabel: string[] = []

    const addHeatmap = (grid: number[][], ramp: string[], opacity: [number, number], max: number, label: string) => {
      const idx = series.length
      series.push({ type: "heatmap", data: gridToHeatmapData(grid), emphasis: { itemStyle: { opacity: 1 } } })
      visualMap.push({ show: false, seriesIndex: idx, min: 0, max: max || 1, inRange: { color: ramp, opacity } })
      seriesLabel[idx] = label
    }
    const addPoi = (grid: number[][], color: string, symbol: string, label: string) => {
      const idx = series.length
      series.push({
        type: "scatter",
        data: gridToPoints(grid),
        symbol,
        symbolSize: 11,
        itemStyle: { color, borderColor: "#fff", borderWidth: 1.25, shadowBlur: 2, shadowColor: "rgba(0,0,0,0.35)" },
      })
      seriesLabel[idx] = label
    }

    for (const key of Z_ORDER) {
      if (!active.has(key)) continue
      const def = layerByKey[key]
      if (!def) continue
      if (def.kind === "infection") addHeatmap(spatialDailyGrids[day], def.ramp!, def.opacity!, zMax, "Cumulative infections")
      else if (def.kind === "poi") addPoi(sl[def.source!], def.swatch, def.symbol!, def.label)
      else addHeatmap(sl[def.source!], def.ramp!, def.opacity!, maxOf(sl[def.source!]), def.label)
    }

    return {
      backgroundColor: "transparent",
      animation: false,
      grid: { left: 0, right: 0, top: 0, bottom: 0 },
      xAxis: { type: "category", data: axes.x.map(String), show: false, boundaryGap: true },
      yAxis: { type: "category", data: axes.y.map(String), show: false, boundaryGap: true },
      visualMap,
      tooltip: {
        position: "top",
        formatter: (p: any) => {
          const label = seriesLabel[p.seriesIndex] ?? ""
          if (p.seriesType === "scatter") return label
          const v = Array.isArray(p.value) ? p.value[2] : undefined
          return `${label}: ${typeof v === "number" ? v.toFixed(1) : v}`
        },
      },
      series,
    } as EChartsOption
    /* eslint-enable @typescript-eslint/no-explicit-any */
  }, [day, active, sl, axes, zMax, spatialDailyGrids, layerByKey])

  useEffect(() => {
    chartRef.current?.setOption(option, true)
  }, [chartRef, option])

  useEffect(() => {
    if (chartRef.current) onReady?.(chartRef.current)
  }, [chartRef, onReady])

  const toggle = (key: string) =>
    setActive((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })

  const gradient = `linear-gradient(to right, ${SEQUENTIAL_BLUE.join(", ")})`
  const showColorbar = active.has("infections")

  return (
    <div className="space-y-3">
      {/* Layer toggles double as the legend (each chip shows its colour). */}
      <div className="flex flex-wrap items-center gap-1.5">
        {layers.map((def) => {
          const on = active.has(def.key)
          return (
            <button
              key={def.key}
              type="button"
              aria-pressed={on}
              onClick={() => toggle(def.key)}
              className={cn(
                "flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors",
                on
                  ? "border-transparent bg-muted font-medium text-foreground"
                  : "border-border text-muted-foreground hover:bg-muted/60",
              )}
            >
              <span
                className={cn("inline-block size-2.5", def.kind === "poi" ? "rounded-full" : "rounded-[3px]")}
                style={{ backgroundColor: def.swatch, opacity: on ? 1 : 0.4 }}
              />
              {def.label}
            </button>
          )
        })}
      </div>

      <div className="relative w-full max-w-md mx-auto" style={{ aspectRatio }}>
        <img
          src="/static/img/akuse_basemap.png"
          alt="Akuse basemap"
          className="absolute inset-0 h-full w-full object-fill rounded-md"
          style={{ filter: "brightness(1.06) saturate(1.05)" }}
        />
        <div ref={containerRef} className="absolute inset-0 h-full w-full" />
      </div>

      {showColorbar && (
        <div className="max-w-md mx-auto space-y-1">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Cumulative infections per cell</span>
            <span className="tabular-nums">0 – {niceCeil(zMax).toLocaleString()}</span>
          </div>
          <div
            className="h-2.5 w-full rounded-full ring-1 ring-black/10"
            style={{ background: gradient }}
            aria-hidden
          />
        </div>
      )}

      <div className="flex items-center gap-3 max-w-md mx-auto">
        <span className="text-xs text-muted-foreground whitespace-nowrap tabular-nums">
          Day {day + 1} / {lastDay + 1}
        </span>
        <Slider
          aria-label="Simulation day"
          value={[day]}
          min={0}
          max={lastDay}
          step={1}
          onValueChange={([newDay]) => setDay(newDay)}
        />
      </div>
    </div>
  )
}

function gridToHeatmapData(grid: number[][]): [number, number, number][] {
  const out: [number, number, number][] = []
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) {
      // Skip empty cells so the satellite basemap shows through.
      if (grid[y][x] > 0) out.push([x, y, grid[y][x]])
    }
  }
  return out
}

function gridToPoints(grid: number[][]): [number, number][] {
  const out: [number, number][] = []
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) {
      if (grid[y][x] > 0) out.push([x, y])
    }
  }
  return out
}

function maxOf(grid: number[][]): number {
  let m = 0
  for (const row of grid) for (const v of row) if (v > m) m = v
  return m > 0 ? m : 1
}

// Round up to 1 significant figure so the colorbar reads "0 – 7,000" instead
// of the raw, arbitrary-looking peak (e.g. 6,969).
function niceCeil(v: number): number {
  if (v <= 0) return 1
  const mag = Math.pow(10, Math.floor(Math.log10(v)))
  return Math.ceil(v / mag) * mag
}
