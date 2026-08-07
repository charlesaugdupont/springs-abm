import { useEffect, useMemo } from "react"
import type { EChartsOption } from "echarts"
import { useECharts } from "@/hooks/useECharts"
import { chartColors, baseGridAxisOption, axisStyle, dayAxisTooltip } from "@/lib/chartTheme"
import type { EChartsLike } from "@/lib/download"

interface SingleSeriesLineChartProps {
  days: number[]
  values: number[]
  yAxisLabel: string
  valueFormatter?: (v: number) => string
  onReady?: (chart: EChartsLike) => void
}

/** Shared by the wealth and care-seeking charts - today's results.js
 * builds near-identical option objects for both (single series, no
 * pathogen split), so one component covers both rather than two
 * near-duplicates. */
export function SingleSeriesLineChart({ days, values, yAxisLabel, valueFormatter, onReady }: SingleSeriesLineChartProps) {
  const { containerRef, chartRef } = useECharts()

  const option = useMemo<EChartsOption>(() => {
    const colors = chartColors()
    const ax = axisStyle(colors)
    return {
      ...baseGridAxisOption(),
      tooltip: {
        trigger: "axis",
        formatter: dayAxisTooltip(valueFormatter ?? ((v) => `${Math.round(v)}`)),
      },
      xAxis: { type: "category", data: days.map(String), name: "Day", nameLocation: "middle", nameGap: 26, ...ax },
      yAxis: {
        type: "value",
        // Short label rotated ALONG the axis (not the default horizontal
        // top-left spot, which clipped) - containLabel reserves room for it.
        name: yAxisLabel,
        nameLocation: "middle",
        nameRotate: 90,
        nameGap: 40,
        nameTextStyle: { color: colors.muted, fontSize: 11 },
        ...ax,
        axisLabel: valueFormatter
          ? { ...ax.axisLabel, formatter: (v: number) => valueFormatter(v) }
          : ax.axisLabel,
      },
      series: [
        {
          type: "line",
          data: values,
          showSymbol: false,
          lineStyle: { width: 2, color: colors.rota },
          itemStyle: { color: colors.rota },
        },
      ],
    }
  }, [days, values, yAxisLabel, valueFormatter])

  useEffect(() => {
    chartRef.current?.setOption(option, true)
  }, [chartRef, option])

  useEffect(() => {
    if (chartRef.current) onReady?.(chartRef.current)
  }, [chartRef, onReady])

  return <div ref={containerRef} className="h-64 w-full" />
}
