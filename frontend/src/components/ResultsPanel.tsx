import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { MetricGrid } from "@/components/MetricCard"
import { PrevalenceLineChart } from "@/components/charts/PrevalenceLineChart"
import { IllnessDaysAreaChart } from "@/components/charts/IllnessDaysAreaChart"
import { SingleSeriesLineChart } from "@/components/charts/SingleSeriesLineChart"
import { SpatialHeatmapChart } from "@/components/charts/SpatialHeatmapChart"
import { useBasemap } from "@/hooks/useBasemap"
import type { SimResultBundle } from "@/api/types"

const PATHOGEN_LABEL: Record<string, string> = { rota: "Rotavirus", campy: "Campylobacter" }

function buildMetrics(result: SimResultBundle): { label: string; value: string }[] {
  const metrics = [
    { label: "Under-5 agents", value: String(result.n_u5) },
    { label: "Runtime", value: `${result.runtime_seconds.toFixed(1)}s` },
    { label: "Ever infected", value: `${(result.proportion_infected_at_least_once * 100).toFixed(1)}%` },
  ]
  for (const name of result.pathogen_names) {
    const peak = result.summary_metrics[`${name}_peak_u5_prevalence`]
    if (typeof peak === "number") {
      metrics.push({ label: `${PATHOGEN_LABEL[name] ?? name} peak U5 prevalence`, value: `${(peak * 100).toFixed(1)}%` })
    }
  }
  const careSeeking = result.summary_metrics.episode_care_seeking_rate
  if (typeof careSeeking === "number") {
    metrics.push({ label: "Episode care-seeking rate", value: `${(careSeeking * 100).toFixed(1)}%` })
  }
  return metrics
}

export function ResultsPanel({ result }: { result: SimResultBundle }) {
  const basemap = useBasemap()

  return (
    <div className="space-y-6">
      <MetricGrid metrics={buildMetrics(result)} />

      <Card data-testid="chart-card-prevalence">
        <CardHeader>
          <CardTitle className="text-base">Prevalence</CardTitle>
        </CardHeader>
        <CardContent>
          <PrevalenceLineChart
            days={result.days}
            u5Prevalence={result.u5_prevalence}
            allAgesPrevalence={result.all_ages_prevalence}
            pathogenNames={result.pathogen_names}
          />
        </CardContent>
      </Card>

      <Card data-testid="chart-card-illness-days">
        <CardHeader>
          <CardTitle className="text-base">Cumulative under-5 illness-days</CardTitle>
        </CardHeader>
        <CardContent>
          <IllnessDaysAreaChart
            days={result.days}
            cumulativeU5IllnessDays={result.cumulative_u5_illness_days}
            pathogenNames={result.pathogen_names}
          />
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Household wealth</CardTitle>
          </CardHeader>
          <CardContent>
            <SingleSeriesLineChart
              days={result.days}
              values={result.mean_household_wealth}
              yAxisLabel="Fraction of max wealth"
            />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Care-seeking events</CardTitle>
          </CardHeader>
          <CardContent>
            <SingleSeriesLineChart
              days={result.days}
              values={result.cumulative_care_seeking_events}
              yAxisLabel="Cumulative events"
            />
          </CardContent>
        </Card>
      </div>

      <Card data-testid="chart-card-spatial">
        <CardHeader>
          <CardTitle className="text-base">Spatial spread</CardTitle>
        </CardHeader>
        <CardContent>
          {basemap.data ? (
            <SpatialHeatmapChart
              basemap={basemap.data}
              spatialDailyGrids={result.spatial_daily_grids}
              gridSize={result.spatial_grid_size}
            />
          ) : (
            <p className="text-sm text-muted-foreground">Loading basemap…</p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
