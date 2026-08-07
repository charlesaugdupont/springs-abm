import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { ParamField } from "./ParamField"
import { EvidenceTierBadge } from "./EvidenceTierBadge"
import { cn } from "@/lib/utils"
import type { ParametersByCategory, ScenarioFormValues } from "@/api/types"

interface CategoryAccordionProps {
  categories: ParametersByCategory[]
  values: ScenarioFormValues
  onChange: (formName: string, value: number | boolean) => void
}

// Per-section accent colors. Full literal class strings (so Tailwind's scanner
// picks them up). `border` = colored left rail, `header` = tinted trigger row,
// `title` = section-name ink, `content` = faint open-panel wash.
interface Accent {
  border: string
  header: string
  title: string
  content: string
}

const CATEGORY_ACCENT: Record<string, Accent> = {
  "Population & Demographics": {
    border: "border-l-violet-400 dark:border-l-violet-500",
    header: "bg-violet-50/70 dark:bg-violet-950/25",
    title: "text-violet-700 dark:text-violet-300",
    content: "bg-violet-50/30 dark:bg-violet-950/10",
  },
  Rotavirus: {
    border: "border-l-blue-400 dark:border-l-blue-500",
    header: "bg-blue-50/70 dark:bg-blue-950/25",
    title: "text-blue-700 dark:text-blue-300",
    content: "bg-blue-50/30 dark:bg-blue-950/10",
  },
  Campylobacter: {
    border: "border-l-orange-400 dark:border-l-orange-500",
    header: "bg-orange-50/70 dark:bg-orange-950/25",
    title: "text-orange-700 dark:text-orange-300",
    content: "bg-orange-50/30 dark:bg-orange-950/10",
  },
  "Illness Mechanics": {
    border: "border-l-rose-400 dark:border-l-rose-500",
    header: "bg-rose-50/70 dark:bg-rose-950/25",
    title: "text-rose-700 dark:text-rose-300",
    content: "bg-rose-50/30 dark:bg-rose-950/10",
  },
  "Care-Seeking & Behavioral Economics": {
    border: "border-l-emerald-400 dark:border-l-emerald-500",
    header: "bg-emerald-50/70 dark:bg-emerald-950/25",
    title: "text-emerald-700 dark:text-emerald-300",
    content: "bg-emerald-50/30 dark:bg-emerald-950/10",
  },
  "Household Economics": {
    border: "border-l-amber-400 dark:border-l-amber-500",
    header: "bg-amber-50/70 dark:bg-amber-950/25",
    title: "text-amber-700 dark:text-amber-300",
    content: "bg-amber-50/30 dark:bg-amber-950/10",
  },
  "Environment, Water & Shocks": {
    border: "border-l-sky-400 dark:border-l-sky-500",
    header: "bg-sky-50/70 dark:bg-sky-950/25",
    title: "text-sky-700 dark:text-sky-300",
    content: "bg-sky-50/30 dark:bg-sky-950/10",
  },
}

const FALLBACK_ACCENT: Accent = {
  border: "border-l-slate-300 dark:border-l-slate-600",
  header: "bg-muted/60",
  title: "text-foreground",
  content: "bg-muted/20",
}

export function CategoryAccordion({ categories, values, onChange }: CategoryAccordionProps) {
  return (
    <Accordion type="multiple" defaultValue={[]} className="gap-3">
      {categories.map((cat) => {
        const accent = CATEGORY_ACCENT[cat.category] ?? FALLBACK_ACCENT
        return (
          <AccordionItem
            key={cat.category}
            value={cat.category}
            className={cn("overflow-hidden rounded-lg border border-l-4", accent.border)}
          >
            <AccordionTrigger
              className={cn(
                "rounded-none px-3 py-3 text-base font-semibold hover:no-underline",
                accent.header,
                accent.title
              )}
            >
              {cat.category}
            </AccordionTrigger>
            <AccordionContent className={cn("px-3 pt-3", accent.content)}>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-5 pt-1">
                {cat.editable.map((meta) => (
                  <ParamField key={meta.path} meta={meta} values={values} onChange={onChange} />
                ))}
              </div>
              {cat.readonly.length > 0 && (
                <div className="mt-4 space-y-2 border-t pt-3">
                  <p className="text-xs font-medium text-muted-foreground">Fixed parameters</p>
                  {cat.readonly.map((meta) => (
                    <div key={meta.path} className="flex items-center justify-between gap-2 text-sm">
                      <span>{meta.label}</span>
                      <EvidenceTierBadge tier={meta.evidence_tier} />
                    </div>
                  ))}
                </div>
              )}
            </AccordionContent>
          </AccordionItem>
        )
      })}
    </Accordion>
  )
}
