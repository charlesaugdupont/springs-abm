import { useEffect, useRef } from "react"
import * as echarts from "echarts/core"
import { CanvasRenderer } from "echarts/renderers"
import { LineChart } from "echarts/charts"
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
} from "echarts/components"

// Tree-shaken registration - only the modules this app actually uses, not
// the "full" echarts bundle. Keeps the shipped JS well under the old
// vendored plotly.min.js's 4.5MB (see the plan's chart-library decision).
echarts.use([
  CanvasRenderer,
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
])

/** Owns one ECharts instance's lifecycle (init/resize/dispose) for the
 * lifetime of the mounted container. Callers own *what* gets drawn - apply
 * an option via the returned chartRef, either declaratively (setOption on
 * every render when the option object changes) or imperatively (targeted
 * partial updates, e.g. the spatial chart's day-scrubber only touching one
 * series' `data`, not the whole option, per the plan's guidance). */
export function useECharts() {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    if (!containerRef.current) return
    // React 19 StrictMode double-invokes effects in dev (mount, cleanup,
    // mount again) to surface missing-cleanup bugs. Guard against a stray
    // still-registered instance on this exact DOM node regardless of cause
    // - echarts.init() on a node that already has one otherwise leaves two
    // overlapping canvases rather than replacing the first.
    const stale = echarts.getInstanceByDom(containerRef.current)
    stale?.dispose()

    const chart = echarts.init(containerRef.current)
    chartRef.current = chart

    const onResize = () => chart.resize()
    window.addEventListener("resize", onResize)

    return () => {
      window.removeEventListener("resize", onResize)
      chart.dispose()
      chartRef.current = null
    }
  }, [])

  return { containerRef, chartRef }
}
