// Stat-tile contract per the dataviz skill: sentence-case label with no
// trailing colon, semibold value in the default proportional figures
// (large standalone numbers should NOT use tabular-nums - that's reserved
// for columns that must align vertically, e.g. table rows).
export function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-xl font-semibold">{value}</p>
    </div>
  )
}

export function MetricGrid({ metrics }: { metrics: { label: string; value: string }[] }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
      {metrics.map((m) => (
        <MetricCard key={m.label} label={m.label} value={m.value} />
      ))}
    </div>
  )
}
