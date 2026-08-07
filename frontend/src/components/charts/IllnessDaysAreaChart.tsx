import { useEffect, useMemo } from "react"
import type { EChartsOption } from "echarts"
import { useECharts } from "@/hooks/useECharts"
import { chartColors, baseGridAxisOption, axisStyle, axisNameStyle, dayAxisTooltip, PATHOGEN_COLOR } from "@/lib/chartTheme"
import type { EChartsLike } from "@/lib/download"

const PATHOGEN_LABEL: Record<string, string> = { rota: "Rotavirus", campy: "Campylobacter" }

interface IllnessDaysAreaChartProps {
  days: number[]
  cumulativeU5IllnessDays: Record<string, number[]>
  pathogenNames: string[]
  onReady?: (chart: EChartsLike) => void
}

export function IllnessDaysAreaChart({ days, cumulativeU5IllnessDays, pathogenNames, onReady }: IllnessDaysAreaChartProps) {
  const { containerRef, chartRef } = useECharts()

  const option = useMemo<EChartsOption>(() => {
    const colors = chartColors()
    const ax = axisStyle(colors)
    const series: EChartsOption["series"] = pathogenNames.map((name) => {
      const hue = colors[PATHOGEN_COLOR[name] ?? "rota"]
      return {
        name: PATHOGEN_LABEL[name] ?? name,
        type: "line",
        data: cumulativeU5IllnessDays[name] ?? [],
        showSymbol: false,
        lineStyle: { width: 2, color: hue },
        itemStyle: { color: hue },
        // ~10% opacity wash, per the mark spec - never a saturated fill.
        areaStyle: { color: hue, opacity: 0.1 },
      }
    })

    return {
      ...baseGridAxisOption(),
      color: [colors.rota, colors.campy],
      legend: { top: 0, textStyle: { color: colors.secondary, fontSize: 11 } },
      tooltip: { trigger: "axis", formatter: dayAxisTooltip((v) => v.toFixed(1)) },
      xAxis: {
        type: "category",
        data: days.map(String),
        name: "Day",
        nameLocation: "middle",
        nameGap: 30,
        nameTextStyle: axisNameStyle(colors),
        ...ax,
      },
      yAxis: {
        type: "value",
        name: "Cumulative illness-days",
        nameLocation: "middle",
        nameRotate: 90,
        nameGap: 44,
        nameTextStyle: axisNameStyle(colors),
        ...ax,
      },
      series,
    }
  }, [days, cumulativeU5IllnessDays, pathogenNames])

  useEffect(() => {
    chartRef.current?.setOption(option, true)
  }, [chartRef, option])

  useEffect(() => {
    if (chartRef.current) onReady?.(chartRef.current)
  }, [chartRef, onReady])

  return <div ref={containerRef} className="h-80 w-full" />
}
