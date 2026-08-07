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

// Muted, tonally-cohesive palette (chosen over the earlier disjointed Tailwind
// hues). Per accent: `border` = mid-hue left rail, `title` = a darker step of
// the same hue (kept >=4.5:1 on the wash), `header`/`content` = faint washes.
// Arbitrary hex values as whole literal strings so Tailwind's scanner keeps them.
const CATEGORY_ACCENT: Record<string, Accent> = {
  "Population & Demographics": {
    border: "border-l-[#5A6BA8]",
    header: "bg-[#5A6BA8]/8",
    title: "text-[#3E4E86]",
    content: "bg-[#5A6BA8]/[0.045]",
  },
  Rotavirus: {
    border: "border-l-[#3B7CB5]",
    header: "bg-[#3B7CB5]/8",
    title: "text-[#2A5E90]",
    content: "bg-[#3B7CB5]/[0.045]",
  },
  Campylobacter: {
    border: "border-l-[#C77A4A]",
    header: "bg-[#C77A4A]/8",
    title: "text-[#9A5528]",
    content: "bg-[#C77A4A]/[0.045]",
  },
  "Illness Mechanics": {
    border: "border-l-[#B15C79]",
    header: "bg-[#B15C79]/8",
    title: "text-[#8A3F5A]",
    content: "bg-[#B15C79]/[0.045]",
  },
  "Care-Seeking & Behavioral Economics": {
    border: "border-l-[#3E9B84]",
    header: "bg-[#3E9B84]/8",
    title: "text-[#2A7261]",
    content: "bg-[#3E9B84]/[0.045]",
  },
  "Household Economics": {
    border: "border-l-[#B08A3E]",
    header: "bg-[#B08A3E]/8",
    title: "text-[#856326]",
    content: "bg-[#B08A3E]/[0.045]",
  },
  "Environment, Water & Shocks": {
    border: "border-l-[#5B93B0]",
    header: "bg-[#5B93B0]/8",
    title: "text-[#3F7089]",
    content: "bg-[#5B93B0]/[0.045]",
  },
}

const FALLBACK_ACCENT: Accent = {
  border: "border-l-slate-300",
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
