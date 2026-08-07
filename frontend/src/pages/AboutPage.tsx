import { useState } from "react"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { EvidenceTierBadge } from "@/components/ParamForm/EvidenceTierBadge"
import { useParameters } from "@/hooks/useParameters"
import { Skeleton } from "@/components/ui/skeleton"
import { cn } from "@/lib/utils"
import {
  Activity,
  Bird,
  ChevronDown,
  Droplets,
  FlaskConical,
  HeartPulse,
  Sparkles,
  Sun,
  Moon,
  Users,
  Wallet,
} from "lucide-react"

/* ------------------------------------------------------------------ *
 * Small presentational helpers - keep the long-form content readable
 * ------------------------------------------------------------------ */

function SectionHeading({
  icon: Icon,
  eyebrow,
  children,
}: {
  icon: React.ComponentType<{ className?: string }>
  eyebrow: string
  children: React.ReactNode
}) {
  return (
    <div className="flex items-start gap-3">
      <span className="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
        <Icon className="size-5" />
      </span>
      <div>
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{eyebrow}</p>
        <h2 className="text-xl font-semibold leading-tight">{children}</h2>
      </div>
    </div>
  )
}

const CALLOUT_TONES = {
  blue: "border-blue-200 bg-blue-50 dark:border-blue-900/70 dark:bg-blue-950/40",
  orange: "border-orange-200 bg-orange-50 dark:border-orange-900/70 dark:bg-orange-950/30",
  emerald: "border-emerald-200 bg-emerald-50 dark:border-emerald-900/70 dark:bg-emerald-950/30",
  amber: "border-amber-200 bg-amber-50 dark:border-amber-900/70 dark:bg-amber-950/30",
  neutral: "border-border bg-muted/40",
} as const

function Callout({
  tone = "neutral",
  title,
  icon: Icon,
  children,
}: {
  tone?: keyof typeof CALLOUT_TONES
  title?: string
  icon?: React.ComponentType<{ className?: string }>
  children: React.ReactNode
}) {
  return (
    <div className={cn("rounded-xl border p-4 text-sm leading-relaxed", CALLOUT_TONES[tone])}>
      {title && (
        <p className="mb-1.5 flex items-center gap-2 font-semibold">
          {Icon && <Icon className="size-4" />}
          {title}
        </p>
      )}
      <div className="space-y-2 text-foreground/90">{children}</div>
    </div>
  )
}

function Equation({ children }: { children: React.ReactNode }) {
  return (
    <div className="my-2 overflow-x-auto rounded-md border border-border bg-background px-3 py-2 font-mono text-[13px] text-foreground">
      {children}
    </div>
  )
}

function TechDetails({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false)
  return (
    <Collapsible
      open={open}
      onOpenChange={setOpen}
      className="my-4 rounded-xl border border-border bg-muted/30"
    >
      <CollapsibleTrigger className="flex w-full items-center justify-between gap-2 px-4 py-3 text-left text-sm font-medium transition-colors hover:bg-muted/60 rounded-xl">
        <span className="flex items-center gap-2">
          <FlaskConical className="size-4 text-muted-foreground" />
          Technical details
        </span>
        <ChevronDown className={cn("size-4 text-muted-foreground transition-transform", open && "rotate-180")} />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="space-y-3 px-4 pb-4 pt-0 text-sm leading-relaxed text-muted-foreground">
          {children}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

function Section({ children }: { children: React.ReactNode }) {
  return <section className="space-y-4">{children}</section>
}

/* ------------------------------------------------------------------ *
 * Collapsed reference: the full parameter list, grouped by evidence tier
 * ------------------------------------------------------------------ */

function ParameterEvidence() {
  const { data, isLoading } = useParameters()
  const [open, setOpen] = useState(false)
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-xl border border-border">
      <CollapsibleTrigger className="flex w-full items-center justify-between gap-2 rounded-xl px-4 py-3 text-left text-sm font-medium transition-colors hover:bg-muted/60">
        <span>All parameters &amp; their evidence</span>
        <ChevronDown className={cn("size-4 text-muted-foreground transition-transform", open && "rotate-180")} />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="space-y-6 px-4 pb-5 pt-1">
          <p className="text-sm text-muted-foreground">
            How much to trust each number: every scenario parameter, grouped by how well-supported it is.
          </p>
          {isLoading && <Skeleton className="h-40 w-full" />}
          {data?.by_evidence_tier.map((tier) => (
            <div key={tier.tier} className="space-y-3">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-semibold">{tier.label}</h3>
                <EvidenceTierBadge tier={tier.tier} />
              </div>
              <ul className="space-y-2">
                {tier.params.map((p) => (
                  <li key={p.path} className="border-b pb-2 text-sm last:border-b-0">
                    <span className="font-medium">{p.label}</span>
                    <p className="text-muted-foreground">{p.rationale}</p>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

/* ------------------------------------------------------------------ */

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-10 space-y-12">
      {/* Hero */}
      <header className="space-y-4">
        <h1 className="text-3xl font-semibold tracking-tight">About the model</h1>
        <p className="text-base leading-relaxed text-muted-foreground">
          <span className="font-semibold text-foreground">SPRINGS ABM</span> is an{" "}
          <span className="font-medium text-foreground">agent-based model</span> of childhood diarrhoeal
          disease in Akuse, Ghana. Instead of tracking averages, it simulates a whole village of
          individual people — living in households, walking to school and the water source, keeping
          animals, falling ill, and deciding whether to seek care — and lets population-level patterns
          emerge from those individual lives. This page explains how it works, from the big picture down
          to the equations (tucked into optional{" "}
          <span className="font-medium text-foreground">Technical details</span> panels for the curious).
        </p>
        <Callout tone="neutral">
          <p>
            The model deliberately couples three things that are usually modelled separately:
          </p>
          <ul className="ml-4 list-disc space-y-1">
            <li>
              <span className="font-medium text-foreground">Epidemic dynamics</span> for two pathogens with
              very different transmission routes;
            </li>
            <li>
              <span className="font-medium text-foreground">Behavioural economics</span> — parents weighing
              the cost and benefit of seeking paid care for a sick child;
            </li>
            <li>
              <span className="font-medium text-foreground">Household economics</span> — wealth and
              health-linked income that feed back into what families can afford.
            </li>
          </ul>
        </Callout>
      </header>

      {/* 1. Two diseases */}
      <Section>
        <SectionHeading icon={Activity} eyebrow="What it simulates">
          Two diseases, side by side
        </SectionHeading>
        <p className="text-sm leading-relaxed text-muted-foreground">
          The model follows two of the leading causes of childhood diarrhoea. They are chosen because they
          spread in almost opposite ways — one mostly person-to-person, the other largely from the
          environment and animals — so a single intervention rarely helps both equally.
        </p>
        <div className="grid gap-4 sm:grid-cols-2">
          <Callout tone="blue" title="Rotavirus" icon={Droplets}>
            <p>A classic contagious virus. It spreads two ways:</p>
            <ul className="ml-4 list-disc space-y-1">
              <li><span className="font-medium text-foreground">Person-to-person</span> contact between people sharing a location;</li>
              <li><span className="font-medium text-foreground">Waterborne</span> — infectious people shed into shared water sources, which then infect others.</li>
            </ul>
          </Callout>
          <Callout tone="orange" title="Campylobacter" icon={Bird}>
            <p>A bacterium tied to animals and food. It has three separate routes:</p>
            <ul className="ml-4 list-disc space-y-1">
              <li><span className="font-medium text-foreground">Zoonotic</span> — exposure from nearby poultry and livestock;</li>
              <li><span className="font-medium text-foreground">Fecal-oral</span> — within-household spread;</li>
              <li><span className="font-medium text-foreground">Food-borne</span> — a steady background risk from contaminated food.</li>
            </ul>
          </Callout>
        </div>
      </Section>

      {/* 2. The world */}
      <Section>
        <SectionHeading icon={Users} eyebrow="The world">
          People, households and places
        </SectionHeading>
        <p className="text-sm leading-relaxed text-muted-foreground">
          The simulation builds a synthetic population modelled on Akuse and places it on a real map grid
          derived from OpenStreetMap. Every person (an <em>agent</em>) belongs to a household and moves
          around a shared landscape of homes, a school, water points, worship and social spots — overlaid
          with a layer of animal density (poultry and livestock) that drives Campylobacter's zoonotic
          route.
        </p>
        <Callout tone="amber" title="Children and adults are not the same" icon={HeartPulse}>
          <p>
            A key modelling choice: only <span className="font-medium text-foreground">children under 5</span>{" "}
            actually fall ill — they get an illness severity, a duration, and a health toll. Adults can
            still be infected, carry the pathogen and pass it on (including shedding into water), but they
            are effectively <span className="font-medium text-foreground">silent, asymptomatic carriers</span>.
            This mirrors the real burden of these diseases falling on the very young.
          </p>
          <p>
            One adult in each household with a child is the <span className="font-medium text-foreground">parent</span> —
            the person who makes every care-seeking decision for that family.
          </p>
        </Callout>
        <TechDetails>
          <p>
            Household size is drawn as <span className="font-mono">1 + Poisson(λ = 2.2)</span> (mean ≈ 3.2),
            with the first member of every multi-person household guaranteed to be an adult; the first adult
            in a household containing at least one child is flagged as the parent.
          </p>
          <p>
            Each parent is assigned one of 32 fixed behavioural personas, sampled by 3-D Latin Hypercube over
            the decision-model parameters (loss aversion, probability distortion, and the weight placed on
            wealth vs. health) — so families differ systematically in how they weigh risk and cost.
          </p>
        </TechDetails>
      </Section>

      {/* 3. A day in the model */}
      <Section>
        <SectionHeading icon={Sun} eyebrow="The clock">
          A day in the model: day phase &amp; night phase
        </SectionHeading>
        <p className="text-sm leading-relaxed text-muted-foreground">
          Time advances one day at a time. Each day has a rhythm that determines who meets whom — and
          therefore where transmission can happen.
        </p>
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 dark:border-amber-900/70 dark:bg-amber-950/30">
            <p className="flex items-center gap-2 text-sm font-semibold"><Sun className="size-4" /> Morning</p>
            <p className="mt-1 text-sm text-foreground/90">Diseases progress: exposed people may become infectious, and the ill may recover.</p>
          </div>
          <div className="rounded-xl border border-blue-200 bg-blue-50 p-4 dark:border-blue-900/70 dark:bg-blue-950/40">
            <p className="flex items-center gap-2 text-sm font-semibold"><Users className="size-4" /> Day phase</p>
            <p className="mt-1 text-sm text-foreground/90">People travel to school, water, worship and social places. Transmission happens between everyone at the same location.</p>
          </div>
          <div className="rounded-xl border border-indigo-200 bg-indigo-50 p-4 dark:border-indigo-900/70 dark:bg-indigo-950/40">
            <p className="flex items-center gap-2 text-sm font-semibold"><Moon className="size-4" /> Night phase</p>
            <p className="mt-1 text-sm text-foreground/90">Everyone returns home. Transmission happens within the household.</p>
          </div>
        </div>
        <p className="text-sm leading-relaxed text-muted-foreground">
          After the night phase, the day's "systems" run in a fixed order — movement, child illness,
          care-seeking, household bookkeeping, the environment (water &amp; animals), and the household
          economy — before the clock ticks over to the next day.
        </p>
        <TechDetails>
          <p>
            Transmission is evaluated <span className="font-medium text-foreground">twice</span> per day (day
            phase at the activity location, night phase at home); disease progression and the behavioural /
            economic systems run <span className="font-medium text-foreground">once</span> per day.
          </p>
          <p>
            The daily system order is load-bearing: Movement → Child illness → Care-seeking → Household →
            Environment → Economy. One consequence is a deliberate one-day lag — a successful care-seek on
            day <span className="font-mono">N</span> only clears the child's illness after that day's illness
            system has already run.
          </p>
        </TechDetails>
      </Section>

      {/* 4. How infection spreads */}
      <Section>
        <SectionHeading icon={Droplets} eyebrow="Transmission">
          How infection spreads
        </SectionHeading>
        <p className="text-sm leading-relaxed text-muted-foreground">
          Whenever susceptible people share a location with infectious ones, each faces a chance of
          catching the pathogen. The more infectious neighbours nearby, the higher the risk — and someone
          who has been infected before is partially protected.
        </p>
        <TechDetails>
          <p>For the contact routes, a susceptible person co-located with <span className="font-mono">k</span> infectious people is infected with probability</p>
          <Equation>{"P(infection) = 1 − (1 − p)^k"}</Equation>
          <p>
            where <span className="font-mono">p = prob_multiplier × base_prob × immunity_factor</span>, and
            prior infections make you harder to reinfect through
          </p>
          <Equation>{"immunity_factor = exp(−κ · N_inf)"}</Equation>
          <p>
            with <span className="font-mono">N_inf</span> the number of times that person has been infected
            before.
          </p>
          <p className="pt-1">
            <span className="font-medium text-foreground">Campylobacter's zoonotic route</span> instead uses an
            exact Beta-Poisson dose-response. The dose comes from local animal density,
          </p>
          <Equation>{"dose = (poultry × w_poultry + ruminant × w_ruminant) × interaction_rate"}</Equation>
          <Equation>{"P(infection | dose) = 1 − ₁F₁(α, α + β, −dose)"}</Equation>
          <p>
            The <span className="font-medium text-foreground">fecal-oral route</span> is a fixed per-contact
            probability among household members when someone at home is infectious; the{" "}
            <span className="font-medium text-foreground">food-borne route</span> is a small flat daily
            probability for everyone, independent of location or anyone else's status.
          </p>
        </TechDetails>
      </Section>

      {/* 5. Getting sick and recovering */}
      <Section>
        <SectionHeading icon={HeartPulse} eyebrow="Disease course">
          Getting sick, and getting better
        </SectionHeading>
        <p className="text-sm leading-relaxed text-muted-foreground">
          Each person moves through familiar disease states: <strong>S</strong>usceptible →
          (<strong>V</strong>accinated) → <strong>E</strong>xposed → <strong>I</strong>nfectious →
          <strong> R</strong>ecovered. After catching a pathogen there is a short latent period before a
          person becomes infectious, then a daily chance of recovering. For a sick child, the model also
          tracks how <em>severe</em> the illness is and how long it lasts.
        </p>
        <Callout tone="emerald" title="Why the disease never fully dies out" icon={Sparkles}>
          <p>
            Immunity in the model is <span className="font-medium text-foreground">leaky</span>: each past
            infection makes the next one milder, but protection never becomes complete. Because the
            population is closed and immunity keeps decaying back toward susceptibility, the diseases settle
            into a low, persistent hum rather than disappearing — an emergent property, not something coded
            in directly.
          </p>
        </Callout>
        <TechDetails>
          <p>
            The latent period is <span className="font-mono">exposure_period</span> days (3 for both pathogens);
            recovery is a daily Bernoulli trial at <span className="font-mono">recovery_rate</span> (≈ 7-day
            illness). Both are literature-grounded and held out of calibration.
          </p>
          <p>A child's illness severity is fixed at onset from age, base severity, and an immunity multiplier</p>
          <Equation>{"clamp( 1 − vaccine_effect − N_inf × per_infection_reduction , min = 0.1 )"}</Equation>
          <p>
            The floor at <span className="font-mono">0.1</span> is exactly why immunity is never perfect — and
            the mechanistic reason the model sustains endemic transmission indefinitely. Severity is then held
            for the whole episode and duration simply counts down.
          </p>
        </TechDetails>
      </Section>

      {/* 6. Seeking care */}
      <Section>
        <SectionHeading icon={Wallet} eyebrow="Behaviour & economics">
          Deciding whether to seek care
        </SectionHeading>
        <p className="text-sm leading-relaxed text-muted-foreground">
          When a child is ill, the parent faces a daily choice: pay to seek care, or wait. Seeking care can
          cure the child — but it costs money the household may not have. Waiting is free but risks the
          illness worsening. The model captures this as a genuine risk-and-reward decision, using{" "}
          <span className="font-medium text-foreground">Cumulative Prospect Theory</span>, the same framework
          behavioural economists use to describe how real people (over-)weigh small probabilities and fear
          losses more than they value equivalent gains.
        </p>
        <div className="grid gap-4 sm:grid-cols-2">
          <Callout tone="emerald" title="Seek care">
            <p>Pay the cost of care. With some success probability the child fully recovers; otherwise the money is spent and the parent still carries the worry.</p>
          </Callout>
          <Callout tone="amber" title="Wait">
            <p>Spend nothing, but the illness may worsen — and the parent bears the stress of a still-sick child.</p>
          </Callout>
        </div>
        <p className="text-sm leading-relaxed text-muted-foreground">
          Crucially, wealth is <span className="font-medium text-foreground">pooled per household</span> and a
          family simply cannot seek care it can't afford. Because successful care also clears the infection,
          higher care-seeking doesn't just shorten illness — it actually lowers how much disease circulates.
        </p>
        <TechDetails>
          <p>The value a parent places on an outcome combines household wealth and an effective health level in a Cobb-Douglas utility:</p>
          <Equation>{"U(w, h_eff) = w^α · h_eff^(1 − α)"}</Equation>
          <Equation>{"h_eff = (1 − child_weight) · h_parent + child_weight · h_child"}</Equation>
          <p>
            Here <span className="font-mono">w</span> is household-pooled wealth and{" "}
            <span className="font-mono">α</span> weights wealth against health (it varies by persona). On top
            of this, the prospect-theory layer applies Prelec probability weighting and a per-parent loss
            aversion <span className="font-mono">λ</span>, with the current sick state as the reference point
            (a cure reads as a gain, a worsening as a loss).
          </p>
          <p>
            Wealth grows through health-linked daily income and is depleted by the cost of care; if a
            household's wealth is below the cost of care, the "could not afford" outcome is recorded. These
            economic parameters were calibrated so the model's care-seeking rate matches Ghana DHS survey data
            (≈ 69%).
          </p>
        </TechDetails>
      </Section>

      {/* 7. What emerges */}
      <Section>
        <SectionHeading icon={Sparkles} eyebrow="Emergent behaviour">
          What comes out of all this
        </SectionHeading>
        <p className="text-sm leading-relaxed text-muted-foreground">
          None of the following is programmed directly — each emerges from the interacting rules above. These
          are the patterns worth watching for when you run a scenario:
        </p>
        <ul className="space-y-3">
          <li className="rounded-lg border border-border bg-muted/30 p-3 text-sm">
            <span className="font-medium text-foreground">One big wave, then a simmer.</span> A dominant
            epidemic wave peaks around day 30 and decays over the following ~150–200 days, settling into a low
            quasi-equilibrium as the pool of susceptible people is depleted.
          </li>
          <li className="rounded-lg border border-border bg-muted/30 p-3 text-sm">
            <span className="font-medium text-foreground">Endemic persistence from leaky immunity.</span> With
            no births in the closed population, it is incomplete immunity — not new susceptibles — that keeps
            the disease alive.
          </li>
          <li className="rounded-lg border border-border bg-muted/30 p-3 text-sm">
            <span className="font-medium text-foreground">Timing beats magnitude for shocks.</span> A water-
            contamination shock does far more damage depending on <em>when</em> it lands relative to the
            outbreak than on how big it is.
          </li>
          <li className="rounded-lg border border-border bg-muted/30 p-3 text-sm">
            <span className="font-medium text-foreground">Vaccine rate and efficacy aren't interchangeable.</span>{" "}
            How well a vaccine works gates whether vaccinating more people helps at all.
          </li>
        </ul>
      </Section>

      {/* 8. Evidence */}
      <Section>
        <SectionHeading icon={FlaskConical} eyebrow="Trust & calibration">
          How much to trust the numbers
        </SectionHeading>
        <p className="text-sm leading-relaxed text-muted-foreground">
          Every parameter carries an evidence label — from{" "}
          <span className="font-medium text-foreground">literature-grounded</span> values taken from published
          studies, through numbers <span className="font-medium text-foreground">calibrated</span> against
          real data, to documented <span className="font-medium text-foreground">assumptions</span> and purely{" "}
          <span className="font-medium text-foreground">structural</span> internals. Expand the list below to
          see every parameter and the reasoning behind its value.
        </p>
        <ParameterEvidence />
      </Section>
    </div>
  )
}
