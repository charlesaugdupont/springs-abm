import { useEffect, useMemo } from "react"
import type { EChartsOption } from "echarts"
import { useECharts } from "@/hooks/useECharts"
import {
  chartColors,
  axisStyle,
  baseGridAxisOption,
  campyRouteColors,
  dayAxisTooltip,
  CAMPY_ROUTE_ORDER,
  CAMPY_ROUTE_LABEL,
} from "@/lib/chartTheme"
import type { EChartsLike } from "@/lib/download"

interface CampyRouteAreaChartProps {
  days: number[]
  /** Per-day NEW Campylobacter infections keyed by route (zoonotic /
   * fecal_oral / food_borne). */
  infectionsByRoute: Record<string, number[]>
  onReady?: (chart: EChartsLike) => void
}

/** 100%-stacked area of Campylobacter's three transmission routes over time.
 * Plots the CUMULATIVE-to-date composition (each route's share of all campy
 * infections so far), not raw daily shares: it's smooth, always defined after
 * the first case (no 0/0), and converges to the run-level route fractions. */
export function CampyRouteAreaChart({ days, infectionsByRoute, onReady }: CampyRouteAreaChartProps) {
  const { containerRef, chartRef } = useECharts()

  const option = useMemo<EChartsOption>(() => {
    const colors = chartColors()
    const ax = axisStyle(colors)
    const routeColor = campyRouteColors()
    const routes = CAMPY_ROUTE_ORDER.filter((r) => infectionsByRoute[r]?.length)
    const n = days.length

    // Cumulative infections-to-date per route, then normalise to a percentage
    // of the running campy total each day.
    const cum: Record<string, number[]> = {}
    for (const r of routes) {
      const daily = infectionsByRoute[r] ?? []
      const acc: number[] = []
      let running = 0
      for (let i = 0; i < n; i++) {
        running += daily[i] ?? 0
        acc.push(running)
      }
      cum[r] = acc
    }
    const proportion: Record<string, number[]> = {}
    for (const r of routes) proportion[r] = new Array(n).fill(0)
    for (let i = 0; i < n; i++) {
      let total = 0
      for (const r of routes) total += cum[r][i]
      if (total > 0) for (const r of routes) proportion[r][i] = (cum[r][i] / total) * 100
    }

    const series: EChartsOption["series"] = routes.map((r) => ({
      name: CAMPY_ROUTE_LABEL[r],
      type: "line",
      stack: "routes",
      data: proportion[r],
      showSymbol: false,
      // 2px surface-colored separator between stacked bands (mark spec).
      lineStyle: { width: 2, color: colors.surface },
      itemStyle: { color: routeColor[r] },
      areaStyle: { color: routeColor[r], opacity: 0.9 },
      // Keep the bands solid at all times - no hover-dimming of the other
      // series (that "focus" emphasis caused the flicker the user reported).
      // Identity comes from the legend + the axis tooltip on hover.
      emphasis: { disabled: true },
    }))

    return {
      ...baseGridAxisOption(),
      grid: { left: 12, right: 20, top: 30, bottom: 40, containLabel: true },
      color: routes.map((r) => routeColor[r]),
      legend: { top: 0, textStyle: { color: colors.secondary, fontSize: 11 } },
      tooltip: {
        trigger: "axis",
        formatter: dayAxisTooltip((v) => `${v.toFixed(1)}%`),
      },
      xAxis: {
        type: "category",
        data: days.map(String),
        name: "Day",
        nameLocation: "middle",
        nameGap: 26,
        boundaryGap: false,
        ...ax,
      },
      yAxis: {
        type: "value",
        min: 0,
        max: 100,
        ...ax,
        axisLabel: { ...ax.axisLabel, formatter: (v: number) => `${v}%` },
      },
      series,
    }
  }, [days, infectionsByRoute])

  useEffect(() => {
    chartRef.current?.setOption(option, true)
  }, [chartRef, option])

  useEffect(() => {
    if (chartRef.current) onReady?.(chartRef.current)
  }, [chartRef, onReady])

  return <div ref={containerRef} className="h-80 w-full" />
}
