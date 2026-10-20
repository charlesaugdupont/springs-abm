import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import type { ScenarioFormValues } from "@/api/types"

// rota_enabled/campy_enabled/crypto_enabled are the one deliberate exception
// to "every field is driven by GET /api/parameters": they're structural
// toggles added directly in webapp/scenario_form.py (gating which OTHER
// fields apply), not scientific parameters, so they were never given
// parameter_registry.py entries at all - not even as category="internal"
// ones. Hardcoding them here mirrors that same distinction, not a gap in the
// data-driven design. Keep this list in step with _PATHOGEN_TOGGLES in
// webapp/scenario_form.py; the server rejects a submission with all of them
// off.
const PATHOGENS: {
  formName: "rota_enabled" | "campy_enabled" | "crypto_enabled"
  label: string
}[] = [
  { formName: "rota_enabled", label: "Rotavirus" },
  { formName: "campy_enabled", label: "Campylobacter" },
  { formName: "crypto_enabled", label: "Cryptosporidium" },
]

interface PathogenToggleProps {
  values: ScenarioFormValues
  onChange: (formName: string, value: boolean) => void
}

export function PathogenToggle({ values, onChange }: PathogenToggleProps) {
  return (
    <div className="flex flex-wrap items-center gap-6">
      {PATHOGENS.map((p) => (
        <div key={p.formName} className="flex items-center gap-2">
          <Switch
            id={p.formName}
            checked={Boolean(values[p.formName] ?? true)}
            onCheckedChange={(checked) => onChange(p.formName, checked)}
          />
          <Label htmlFor={p.formName}>{p.label}</Label>
        </div>
      ))}
    </div>
  )
}
