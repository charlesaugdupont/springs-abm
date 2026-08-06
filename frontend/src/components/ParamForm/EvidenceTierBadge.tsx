import { Badge } from "@/components/ui/badge"
import type { EvidenceTier } from "@/api/types"
import { cn } from "@/lib/utils"

const TIER_LABELS: Record<EvidenceTier, string> = {
  literature: "Literature-grounded",
  calibrated: "Calibrated against data",
  assumption: "Documented assumption",
  structural: "Structural / internal",
}

// Same four-tier meaning as the old .tier-* CSS classes (style.css), ported
// to Tailwind/shadcn tokens rather than bespoke hex values.
const TIER_STYLES: Record<EvidenceTier, string> = {
  literature: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  calibrated: "bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300",
  assumption: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  structural: "bg-muted text-muted-foreground",
}

export function EvidenceTierBadge({ tier, className }: { tier: EvidenceTier; className?: string }) {
  return (
    <Badge variant="secondary" className={cn(TIER_STYLES[tier], className)}>
      {TIER_LABELS[tier]}
    </Badge>
  )
}
