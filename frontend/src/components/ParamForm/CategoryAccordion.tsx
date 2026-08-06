import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { ParamField } from "./ParamField"
import { EvidenceTierBadge } from "./EvidenceTierBadge"
import type { ParametersByCategory, ScenarioFormValues } from "@/api/types"

interface CategoryAccordionProps {
  categories: ParametersByCategory[]
  values: ScenarioFormValues
  onChange: (formName: string, value: number | boolean) => void
}

export function CategoryAccordion({ categories, values, onChange }: CategoryAccordionProps) {
  return (
    <Accordion type="multiple" defaultValue={categories.map((c) => c.category)}>
      {categories.map((cat) => (
        <AccordionItem key={cat.category} value={cat.category}>
          <AccordionTrigger>
            {cat.category}
            <span className="ml-2 text-xs font-normal text-muted-foreground">
              {cat.editable.length} parameter{cat.editable.length === 1 ? "" : "s"}
            </span>
          </AccordionTrigger>
          <AccordionContent>
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
      ))}
    </Accordion>
  )
}
