import { useParameters } from "@/hooks/useParameters"
import { EvidenceTierBadge } from "@/components/ParamForm/EvidenceTierBadge"
import { Skeleton } from "@/components/ui/skeleton"

export default function AboutPage() {
  const { data, isLoading } = useParameters()

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">About the model</h1>
        <p className="text-muted-foreground text-sm mt-1">
          How much to trust each number: every scenario parameter, grouped by evidence tier.
        </p>
      </header>

      {isLoading && <Skeleton className="h-64 w-full" />}

      {data?.by_evidence_tier.map((tier) => (
        <section key={tier.tier} className="space-y-3">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-medium">{tier.label}</h2>
            <EvidenceTierBadge tier={tier.tier} />
          </div>
          <ul className="space-y-2">
            {tier.params.map((p) => (
              <li key={p.path} className="text-sm border-b pb-2">
                <span className="font-medium">{p.label}</span>
                <p className="text-muted-foreground">{p.rationale}</p>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  )
}
