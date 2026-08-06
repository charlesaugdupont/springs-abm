import { useEffect, useMemo } from "react"
import type { EChartsOption } from "echarts"
import { useECharts } from "@/hooks/useECharts"
import { chartColors, baseGridAxisOption } from "@/lib/chartTheme"

interface SingleSeriesLineChartProps {
  days: number[]
  values: number[]
  yAxisLabel: string
  valueFormatter?: (v: number) => string
}

/** Shared by the wealth and care-seeking charts - today's results.js
 * builds near-identical option objects for both (single series, no
 * pathogen split), so one component covers both rather than two
 * near-duplicates. */
export function SingleSeriesLineChart({ days, values, yAxisLabel, valueFormatter }: SingleSeriesLineChartProps) {
  const { containerRef, chartRef } = useECharts()

  const option = useMemo<EChartsOption>(() => {
    const colors = chartColors()
    return {
      ...baseGridAxisOption(colors),
      tooltip: {
        trigger: "axis",
        valueFormatter: valueFormatter ? (v) => valueFormatter(v as number) : undefined,
      },
      xAxis: { type: "category", data: days.map(String), name: "Day", nameLocation: "middle", nameGap: 24 },
      yAxis: {
        type: "value",
        name: yAxisLabel,
        axisLabel: valueFormatter ? { formatter: (v: number) => valueFormatter(v) } : undefined,
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

  return <div ref={containerRef} className="h-64 w-full" />
}
