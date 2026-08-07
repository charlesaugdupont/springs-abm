import { useEffect, useMemo, useState } from "react"
import type { EChartsOption } from "echarts"
import { useECharts } from "@/hooks/useECharts"
import { SEQUENTIAL_BLUE } from "@/lib/chartTheme"
import { spatialAxes, type BasemapMeta } from "@/lib/spatialAxes"
import { Slider } from "@/components/ui/slider"

interface SpatialHeatmapChartProps {
  basemap: BasemapMeta
  spatialDailyGrids: number[][][]
  gridSize: number
}

/** Cumulative infection-density heatmap over the Akuse basemap, with a
 * day-by-day scrubber. The background image is a plain <img>, absolutely
 * positioned behind a transparent-background ECharts canvas, both sized to
 * the same container (itself CSS aspect-ratio'd to the basemap's true
 * geographic extent - the basemap PNG is a SQUARE 1000x1000 render of a
 * non-square lon/lat box, so it must be stretched, not shown at its native
 * aspect ratio - see scripts/bake_basemap.py). This is simpler and more
 * robust than ECharts `graphic` image elements since this chart has no
 * zoom/pan - a fixed 25x25 grid over a fixed extent needs no coordinate
 * recompute-on-resize logic. The day scrubber updates via a targeted
 * `setOption({series:[{data}]})` call, not a full option rebuild, since
 * dragging can fire many times a second across up to ~500 days. */
export function SpatialHeatmapChart({ basemap, spatialDailyGrids, gridSize }: SpatialHeatmapChartProps) {
  const { containerRef, chartRef } = useECharts()
  const lastDay = spatialDailyGrids.length - 1
  const [day, setDay] = useState(lastDay) // default to the final day - the fullest picture

  const axes = useMemo(() => spatialAxes(basemap, gridSize), [basemap, gridSize])
  const aspectRatio = (basemap.maxx - basemap.minx) / (basemap.maxy - basemap.miny)

  const zMax = useMemo(() => {
    let max = 0
    for (const grid of spatialDailyGrids) for (const row of grid) for (const v of row) if (v > max) max = v
    return max > 0 ? max : 1
  }, [spatialDailyGrids])

  const initialOption = useMemo<EChartsOption>(() => {
    return {
      backgroundColor: "transparent",
      animation: false,
      grid: { left: 0, right: 0, top: 0, bottom: 0 },
      xAxis: { type: "category", data: axes.x.map(String), show: false, boundaryGap: true },
      yAxis: { type: "category", data: axes.y.map(String), show: false, boundaryGap: true },
      visualMap: {
        min: 0,
        max: zMax,
        // Hidden: the scale is drawn as a labelled HTML colorbar BELOW the map
        // instead (was overlaid on the image with no numbers). The object stays
        // so it still drives the cell color + opacity mapping.
        show: false,
        // Low-value cells stay translucent so the satellite basemap shows
        // through; only the busiest cells approach opaque. This, plus dropping
        // zero cells entirely (gridToHeatmapData), stops the heatmap from
        // blanketing the whole image.
        inRange: { color: SEQUENTIAL_BLUE, opacity: [0.3, 0.82] },
      },
      tooltip: {
        position: "top",
        formatter: (p) => {
          const params = p as unknown as { value: [number, number, number] }
          return `Cumulative infections: ${params.value[2].toFixed(1)}`
        },
      },
      series: [
        {
          type: "heatmap",
          data: gridToHeatmapData(spatialDailyGrids[lastDay]),
          emphasis: { itemStyle: { opacity: 1 } },
        },
      ],
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }
  }, [axes, zMax, lastDay, spatialDailyGrids])

  useEffect(() => {
    chartRef.current?.setOption(initialOption, true)
  }, [chartRef, initialOption])

  const handleDayChange = ([newDay]: number[]) => {
    setDay(newDay)
    chartRef.current?.setOption({
      series: [{ data: gridToHeatmapData(spatialDailyGrids[newDay]) }],
    })
  }

  const gradient = `linear-gradient(to right, ${SEQUENTIAL_BLUE.join(", ")})`

  return (
    <div className="space-y-3">
      <div className="relative w-full max-w-md mx-auto" style={{ aspectRatio }}>
        <img
          src="/static/img/akuse_basemap.png"
          alt="Akuse basemap"
          className="absolute inset-0 h-full w-full object-fill rounded-md"
          style={{ filter: "brightness(1.06) saturate(1.05)" }}
        />
        <div ref={containerRef} className="absolute inset-0 h-full w-full" />
      </div>

      {/* Labelled colorbar, below the map (previously overlaid on it with no
          numbers). Hue encodes cumulative infections per cell. */}
      <div className="max-w-md mx-auto space-y-1">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Cumulative infections per cell</span>
          <span className="tabular-nums">
            0 – {Math.round(zMax).toLocaleString()}
          </span>
        </div>
        <div
          className="h-2.5 w-full rounded-full ring-1 ring-foreground/10"
          style={{ background: gradient }}
          aria-hidden
        />
        <div className="flex items-center justify-between text-[10px] text-muted-foreground tabular-nums">
          <span>fewer</span>
          <span>more</span>
        </div>
      </div>
      <div className="flex items-center gap-3 max-w-md mx-auto">
        <span className="text-xs text-muted-foreground whitespace-nowrap">Day {day}</span>
        <Slider
          aria-label="Simulation day"
          value={[day]}
          min={0}
          max={lastDay}
          step={1}
          onValueChange={handleDayChange}
        />
      </div>
    </div>
  )
}

function gridToHeatmapData(grid: number[][]): [number, number, number][] {
  const out: [number, number, number][] = []
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) {
      // Skip empty cells entirely - painting every zero-value cell (at the
      // lightest blue + a flat 0.7 opacity) is what blanketed the satellite
      // basemap. Only occupied cells are drawn now.
      if (grid[y][x] > 0) out.push([x, y, grid[y][x]])
    }
  }
  return out
}
