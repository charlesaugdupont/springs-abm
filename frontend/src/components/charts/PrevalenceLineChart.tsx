import { useEffect, useMemo } from "react"
import type { EChartsOption } from "echarts"
import { useECharts } from "@/hooks/useECharts"
import { chartColors, baseGridAxisOption, axisStyle, PATHOGEN_COLOR } from "@/lib/chartTheme"

const PATHOGEN_LABEL: Record<string, string> = { rota: "Rotavirus", campy: "Campylobacter" }

interface PrevalenceLineChartProps {
  days: number[]
  u5Prevalence: Record<string, number[]>
  allAgesPrevalence: Record<string, number[]>
  pathogenNames: string[]
}

export function PrevalenceLineChart({ days, u5Prevalence, allAgesPrevalence, pathogenNames }: PrevalenceLineChartProps) {
  const { containerRef, chartRef } = useECharts()

  const option = useMemo<EChartsOption>(() => {
    const colors = chartColors()
    const ax = axisStyle(colors)
    const series: EChartsOption["series"] = []
    for (const name of pathogenNames) {
      const hue = colors[PATHOGEN_COLOR[name] ?? "rota"]
      series.push({
        name: `${PATHOGEN_LABEL[name] ?? name} - under 5`,
        type: "line",
        data: u5Prevalence[name] ?? [],
        showSymbol: false,
        lineStyle: { width: 2, color: hue, type: "solid" },
        itemStyle: { color: hue },
      })
      series.push({
        name: `${PATHOGEN_LABEL[name] ?? name} - all ages`,
        type: "line",
        data: allAgesPrevalence[name] ?? [],
        showSymbol: false,
        lineStyle: { width: 2, color: hue, type: "dashed" },
        itemStyle: { color: hue },
      })
    }

    return {
      ...baseGridAxisOption(),
      color: [colors.rota, colors.campy],
      legend: { top: 0, textStyle: { color: colors.secondary, fontSize: 11 } },
      tooltip: {
        trigger: "axis",
        valueFormatter: (v) => `${((v as number) * 100).toFixed(2)}%`,
      },
      xAxis: { type: "category", data: days.map(String), name: "Day", nameLocation: "middle", nameGap: 26, ...ax },
      yAxis: {
        type: "value",
        ...ax,
        axisLabel: { ...ax.axisLabel, formatter: (v: number) => `${(v * 100).toFixed(0)}%` },
      },
      series,
    }
  }, [days, u5Prevalence, allAgesPrevalence, pathogenNames])

  useEffect(() => {
    chartRef.current?.setOption(option, true)
  }, [chartRef, option])

  return <div ref={containerRef} className="h-80 w-full" />
}
