import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import type { MutableRefObject } from "react"
import { SEQUENTIAL_BLUE } from "@/lib/chartTheme"
import { type BasemapMeta } from "@/lib/spatialAxes"
import { Slider } from "@/components/ui/slider"
import { cn } from "@/lib/utils"
import { downloadBlob, loadImage } from "@/lib/download"

interface SpatialHeatmapChartProps {
  basemap: BasemapMeta
  spatialDailyGrids: number[][][]
  gridSize: number
  staticLayers?: Record<string, number[][]>
  /** Populated with a composite-PNG exporter so the card header button can call it. */
  exportRef?: MutableRefObject<(() => void) | null>
}

const BASEMAP_SRC = "/static/img/akuse_basemap.png"
const BASEMAP_FILTER = "brightness(1.06) saturate(1.05)"

// Single-hue ramps for the density base layers (distinct from infection blue).
const GREY_RAMP = ["#e4e4e7", "#a1a1aa", "#3f3f46"]
const AMBER_RAMP = ["#fde68a", "#f59e0b", "#b45309"]
const RIVER_RAMP = ["#bcd8f5", "#3b82f6"]

interface BaseDef {
  key: string
  label: string
  swatch: string
  kind: "none" | "density"
  source?: string // key in staticLayers ("infections" pulls from the day grids)
  ramp?: string[]
  opacity?: [number, number]
  colorbar?: { label: string; max: number } | null
}
interface PoiDef {
  key: string
  label: string
  swatch: string
  source: string
  symbol: "circle" | "rect" | "diamond"
}

/** Smooth (bilinearly upscaled) infection/density map over the Akuse basemap,
 * with a day scrubber and toggleable reference overlays. Rendered on a plain
 * canvas (not ECharts) so the density layer interpolates into a continuous
 * field instead of hard squares. Base layers are mutually exclusive; POI point
 * layers overlay on top. */
export function SpatialHeatmapChart({ basemap, spatialDailyGrids, staticLayers, exportRef }: SpatialHeatmapChartProps) {
  const sl = useMemo(() => staticLayers ?? {}, [staticLayers])
  const lastDay = spatialDailyGrids.length - 1
  const [day, setDay] = useState(lastDay)
  const [base, setBase] = useState("infections")
  const [pois, setPois] = useState<Set<string>>(() => new Set())

  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const aspectRatio = (basemap.maxx - basemap.minx) / (basemap.maxy - basemap.miny)

  const zMax = useMemo(() => {
    let m = 0
    for (const g of spatialDailyGrids) for (const row of g) for (const v of row) if (v > m) m = v
    return m > 0 ? m : 1
  }, [spatialDailyGrids])

  const baseLayers = useMemo<BaseDef[]>(() => {
    const defs: BaseDef[] = [
      { key: "natural", label: "Natural", swatch: "#3dd498", kind: "none", colorbar: null },
      { key: "infections", label: "Infections", swatch: "#256abf", kind: "density", source: "infections", ramp: SEQUENTIAL_BLUE, opacity: [0.25, 0.85], colorbar: { label: "Cumulative infections per cell", max: zMax } },
    ]
    if (sl.household_density) defs.push({ key: "households", label: "Households", swatch: "#52525b", kind: "density", source: "household_density", ramp: GREY_RAMP, opacity: [0.3, 0.85], colorbar: { label: "Households per cell", max: maxOf(sl.household_density) } })
    if (sl.animal_density) defs.push({ key: "animals", label: "Animals", swatch: "#b5701a", kind: "density", source: "animal_density", ramp: AMBER_RAMP, opacity: [0.2, 0.8], colorbar: { label: "Animal density (relative)", max: maxOf(sl.animal_density) } })
    if (sl.natural_water) defs.push({ key: "rivers", label: "Rivers", swatch: "#5b9bd8", kind: "density", source: "natural_water", ramp: RIVER_RAMP, opacity: [0.35, 0.6], colorbar: null })
    return defs
  }, [sl, zMax])
  const baseByKey = useMemo(() => Object.fromEntries(baseLayers.map((d) => [d.key, d])), [baseLayers])

  const poiLayers = useMemo<PoiDef[]>(() => {
    const defs: PoiDef[] = []
    if (sl.school) defs.push({ key: "schools", label: "Schools", swatch: "#6d5ac0", source: "school", symbol: "rect" })
    if (sl.water) defs.push({ key: "water", label: "Water points", swatch: "#2f80c4", source: "water", symbol: "circle" })
    if (sl.place_of_worship) defs.push({ key: "worship", label: "Worship", swatch: "#b8763a", source: "place_of_worship", symbol: "diamond" })
    return defs
  }, [sl])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return
    const w = container.clientWidth
    const h = container.clientHeight
    if (w === 0 || h === 0) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(w * dpr)
    canvas.height = Math.round(h * dpr)
    canvas.style.width = `${w}px`
    canvas.style.height = `${h}px`
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)

    const baseDef = baseByKey[base]
    if (baseDef?.kind === "density") {
      const grid = baseDef.key === "infections" ? spatialDailyGrids[day] : sl[baseDef.source!]
      if (grid) drawSmoothDensity(ctx, grid, baseDef.ramp!, baseDef.opacity!, w, h)
    }
    for (const def of poiLayers) {
      if (pois.has(def.key) && sl[def.source]) drawPois(ctx, sl[def.source], def, w, h)
    }
  }, [base, day, pois, sl, spatialDailyGrids, baseByKey, poiLayers])

  useEffect(() => {
    draw()
  }, [draw])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const ro = new ResizeObserver(() => draw())
    ro.observe(container)
    return () => ro.disconnect()
  }, [draw])

  // Composite export: basemap + overlay canvas -> PNG. Reads the live canvas, so
  // it always reflects the current layers/day.
  useEffect(() => {
    if (!exportRef) return
    exportRef.current = async () => {
      const canvas = canvasRef.current
      if (!canvas) return
      const out = document.createElement("canvas")
      out.width = canvas.width
      out.height = canvas.height
      const octx = out.getContext("2d")
      if (!octx) return
      try {
        const baseImg = await loadImage(BASEMAP_SRC)
        octx.filter = BASEMAP_FILTER
        octx.drawImage(baseImg, 0, 0, out.width, out.height)
        octx.filter = "none"
      } catch {
        // basemap failed to load - still export the overlay
      }
      octx.drawImage(canvas, 0, 0, out.width, out.height)
      out.toBlob((blob) => {
        if (blob) downloadBlob("spatial-spread.png", blob)
      }, "image/png")
    }
    return () => {
      if (exportRef) exportRef.current = null
    }
  }, [exportRef])

  const togglePoi = (key: string) =>
    setPois((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })

  const colorbar = baseByKey[base]?.colorbar
  const colorbarRamp = baseByKey[base]?.ramp ?? SEQUENTIAL_BLUE

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        <span className="mr-1 text-muted-foreground">View:</span>
        {baseLayers.map((def) => (
          <Chip key={def.key} label={def.label} swatch={def.swatch} shape="square" active={base === def.key} onClick={() => setBase(def.key)} />
        ))}
      </div>
      {poiLayers.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <span className="mr-1 text-muted-foreground">Points:</span>
          {poiLayers.map((def) => (
            <Chip key={def.key} label={def.label} swatch={def.swatch} shape="dot" active={pois.has(def.key)} onClick={() => togglePoi(def.key)} />
          ))}
        </div>
      )}

      <div ref={containerRef} className="relative w-full max-w-md mx-auto" style={{ aspectRatio }}>
        <img
          src={BASEMAP_SRC}
          alt="Akuse basemap"
          className="absolute inset-0 h-full w-full object-fill rounded-md"
          style={{ filter: BASEMAP_FILTER }}
        />
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full rounded-md" />
      </div>

      {colorbar && (
        <div className="max-w-md mx-auto space-y-1">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>{colorbar.label}</span>
            <span className="tabular-nums">0 – {niceCeil(colorbar.max).toLocaleString()}</span>
          </div>
          <div
            className="h-2.5 w-full rounded-full ring-1 ring-black/10"
            style={{ background: `linear-gradient(to right, ${colorbarRamp.join(", ")})` }}
            aria-hidden
          />
        </div>
      )}

      {base === "infections" && (
        <div className="flex items-center gap-3 max-w-md mx-auto">
          <span className="text-xs text-muted-foreground whitespace-nowrap tabular-nums">
            Day {day + 1} / {lastDay + 1}
          </span>
          <Slider aria-label="Simulation day" value={[day]} min={0} max={lastDay} step={1} onValueChange={([d]) => setDay(d)} />
        </div>
      )}
    </div>
  )
}

function Chip({ label, swatch, shape, active, onClick }: { label: string; swatch: string; shape: "square" | "dot"; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className={cn(
        "flex items-center gap-1.5 rounded-full border px-2.5 py-1 transition-colors",
        active ? "border-transparent bg-muted font-medium text-foreground" : "border-border text-muted-foreground hover:bg-muted/60",
      )}
    >
      <span
        className={cn("inline-block size-2.5", shape === "dot" ? "rounded-full" : "rounded-[3px]")}
        style={{ backgroundColor: swatch, opacity: active ? 1 : 0.45 }}
      />
      {label}
    </button>
  )
}

/* ---- canvas rendering helpers ---- */

function drawSmoothDensity(ctx: CanvasRenderingContext2D, grid: number[][], ramp: string[], opacity: [number, number], w: number, h: number) {
  const n = grid.length
  if (n === 0) return
  const maxV = maxOf(grid)
  const off = document.createElement("canvas")
  off.width = n
  off.height = n
  const octx = off.getContext("2d")
  if (!octx) return
  const img = octx.createImageData(n, n)
  for (let y = 0; y < n; y++) {
    const row = grid[y]
    for (let x = 0; x < row.length; x++) {
      const v = row[x]
      const t = maxV > 0 ? v / maxV : 0
      const [r, g, b] = rampColor(ramp, t)
      const a = v > 0 ? opacity[0] + (opacity[1] - opacity[0]) * t : 0
      // Flip Y: grid y=0 is south (miny) -> bottom row of the (north-up) image.
      const idx = ((n - 1 - y) * n + x) * 4
      img.data[idx] = r
      img.data[idx + 1] = g
      img.data[idx + 2] = b
      img.data[idx + 3] = Math.round(a * 255)
    }
  }
  octx.putImageData(img, 0, 0)
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = "high"
  ctx.drawImage(off, 0, 0, n, n, 0, 0, w, h)
}

function drawPois(ctx: CanvasRenderingContext2D, grid: number[][], def: PoiDef, w: number, h: number) {
  const n = grid.length
  const cw = w / n
  const ch = h / n
  ctx.save()
  ctx.fillStyle = def.swatch
  ctx.strokeStyle = "#ffffff"
  ctx.lineWidth = 1.25
  ctx.shadowColor = "rgba(0,0,0,0.35)"
  ctx.shadowBlur = 2
  const r = 5
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < grid[y].length; x++) {
      if (grid[y][x] <= 0) continue
      const px = (x + 0.5) * cw
      const py = (n - 1 - y + 0.5) * ch
      ctx.beginPath()
      if (def.symbol === "circle") {
        ctx.arc(px, py, r, 0, Math.PI * 2)
      } else if (def.symbol === "rect") {
        ctx.rect(px - r, py - r, r * 2, r * 2)
      } else {
        ctx.moveTo(px, py - r)
        ctx.lineTo(px + r, py)
        ctx.lineTo(px, py + r)
        ctx.lineTo(px - r, py)
        ctx.closePath()
      }
      ctx.fill()
      ctx.stroke()
    }
  }
  ctx.restore()
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "")
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]
}

function rampColor(ramp: string[], t: number): [number, number, number] {
  const tt = Math.max(0, Math.min(1, t))
  const seg = tt * (ramp.length - 1)
  const i = Math.floor(seg)
  const f = seg - i
  const c0 = hexToRgb(ramp[i])
  const c1 = hexToRgb(ramp[Math.min(i + 1, ramp.length - 1)])
  return [
    Math.round(c0[0] + (c1[0] - c0[0]) * f),
    Math.round(c0[1] + (c1[1] - c0[1]) * f),
    Math.round(c0[2] + (c1[2] - c0[2]) * f),
  ]
}

function maxOf(grid: number[][]): number {
  let m = 0
  for (const row of grid) for (const v of row) if (v > m) m = v
  return m > 0 ? m : 1
}

// Round up to 1 significant figure so the colorbar reads "0 – 7,000".
function niceCeil(v: number): number {
  if (v <= 0) return 1
  const mag = Math.pow(10, Math.floor(Math.log10(v)))
  return Math.ceil(v / mag) * mag
}
