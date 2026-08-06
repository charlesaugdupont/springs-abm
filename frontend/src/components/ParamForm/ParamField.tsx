import { Info, Shuffle } from "lucide-react"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { ParamMeta, ScenarioFormValues } from "@/api/types"
import { formatNumber, defaultStep } from "@/lib/formatNumber"

// The one thing every widget variant below shares: a label with an info
// button whose tooltip (Radix, via shadcn) renders in a portal with
// built-in collision detection - the actual fix for the old tooltip-
// clipping bug (an `overflow:hidden` ancestor clipping a plain absolutely-
// positioned <div>, with no viewport-edge awareness at all).
function FieldLabel({ meta }: { meta: ParamMeta }) {
  return (
    <div className="flex items-center gap-1.5">
      <Label>{meta.label}</Label>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            aria-label={`More info about ${meta.label}`}
            className="text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-full"
          >
            <Info className="size-3.5" />
          </button>
        </TooltipTrigger>
        <TooltipContent>
          <p>{meta.rationale}</p>
        </TooltipContent>
      </Tooltip>
    </div>
  )
}

interface ParamFieldProps {
  meta: ParamMeta
  values: ScenarioFormValues
  onChange: (formName: string, value: number | boolean) => void
}

export function ParamField({ meta, values, onChange }: ParamFieldProps) {
  if (!meta.editable || !meta.form_name) return null

  if (meta.ui_widget === "range-pair") {
    return <RangePairField meta={meta} values={values} onChange={onChange} />
  }

  if (meta.ui_widget === "number+randomize-button") {
    return <NumberRandomizeField meta={meta} values={values} onChange={onChange} />
  }

  return <SliderField meta={meta} values={values} onChange={onChange} />
}

function SliderField({ meta, values, onChange }: ParamFieldProps) {
  const formName = meta.form_name!
  const min = meta.ui_min ?? 0
  const max = meta.ui_max ?? 1
  const step = meta.ui_step ?? defaultStep(min, max)
  const value = typeof values[formName] === "number" ? (values[formName] as number) : min

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <FieldLabel meta={meta} />
        <span className="text-sm text-muted-foreground tabular-nums whitespace-nowrap">
          {formatNumber(value, meta.is_integer)}
          {meta.unit ? ` ${meta.unit}` : ""}
        </span>
      </div>
      <Slider
        aria-label={meta.label}
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={([v]) => onChange(formName, meta.is_integer ? Math.round(v) : v)}
      />
    </div>
  )
}

function NumberRandomizeField({ meta, values, onChange }: ParamFieldProps) {
  const formName = meta.form_name!
  const min = meta.ui_min ?? 0
  const max = meta.ui_max ?? Number.MAX_SAFE_INTEGER
  const value = typeof values[formName] === "number" ? (values[formName] as number) : min

  const randomize = () => {
    const next = Math.floor(Math.random() * (max - min + 1)) + min
    onChange(formName, next)
  }

  return (
    <div className="space-y-2">
      <FieldLabel meta={meta} />
      <div className="flex items-center gap-2">
        <Input
          type="number"
          aria-label={meta.label}
          min={min}
          max={max}
          value={value}
          onChange={(e) => {
            const n = Number(e.target.value)
            if (!Number.isNaN(n)) onChange(formName, n)
          }}
          className="max-w-[12rem]"
        />
        <Button type="button" variant="outline" size="icon" onClick={randomize} aria-label={`Randomize ${meta.label}`}>
          <Shuffle className="size-4" />
        </Button>
      </div>
    </div>
  )
}

function RangePairField({ meta, values, onChange }: ParamFieldProps) {
  const formName = meta.form_name!
  const min = meta.ui_min ?? 0
  const max = meta.ui_max ?? 1
  const loKey = `${formName}_min`
  const hiKey = `${formName}_max`
  const lo = typeof values[loKey] === "number" ? (values[loKey] as number) : min
  const hi = typeof values[hiKey] === "number" ? (values[hiKey] as number) : max

  return (
    <div className="space-y-2">
      <FieldLabel meta={meta} />
      <div className="flex items-center gap-2">
        <Input
          type="number"
          min={min}
          max={max}
          value={lo}
          onChange={(e) => {
            const n = Number(e.target.value)
            if (!Number.isNaN(n)) onChange(loKey, n)
          }}
        />
        <span className="text-muted-foreground text-sm">to</span>
        <Input
          type="number"
          min={min}
          max={max}
          value={hi}
          onChange={(e) => {
            const n = Number(e.target.value)
            if (!Number.isNaN(n)) onChange(hiKey, n)
          }}
        />
      </div>
    </div>
  )
}
