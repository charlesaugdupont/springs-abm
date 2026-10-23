"""
Rotavirus natural history from cohorts with denser sampling than MAL-ED -
the anchors for TRUE infection in the model, which MAL-ED's monthly
surveillance under-counts (see experiments/observation.py).

Velazquez FR, Matson DO, Calva JJ, et al. Rotavirus infection in infants as
protection against subsequent infections. N Engl J Med 1996;335:1022-8.
Mexico City, 200 infants followed from birth to 2 years, weekly stools and
serology. Adjusted relative risks after 1, 2, 3 previous infections:
any infection 0.62, 0.40, 0.34; rotavirus diarrhoea 0.23, 0.17, 0.08.

Gladstone BP, Ramani S, Mukhopadhya I, et al. Protective effect of natural
rotavirus infection in an Indian birth cohort. N Engl J Med 2011;365:337-46.
Vellore, 452 children recruited at birth (373 completed 3 years), stool every
2 weeks and during diarrhoea, serum every 6 months. 56% infected by 6 months
of age; ~30% of identified infections were primary (reinfection common);
protection against moderate or severe disease 79% after three infections.

Rogawski McQuade ET, et al. Protection from natural immunity against enteric
infections and etiology-specific diarrhea in a longitudinal birth cohort.
J Infect Dis 2020;222:1858-68 (MAL-ED). Calibrated hazard ratio of
rotavirus diarrhoea after prior rotavirus infection 0.50 (0.41-0.62);
Cryptosporidium diarrhoea 0.32 (0.20-0.51); minimal protection for the
bacteria, including Campylobacter.
"""

#: Relative risk after 1, 2, 3 previous infections (Velazquez et al. 1996).
VELAZQUEZ_RR_INFECTION = (0.62, 0.40, 0.34)
VELAZQUEZ_RR_DIARRHOEA = (0.23, 0.17, 0.08)

#: Gladstone et al. 2011 (Vellore): fraction infected by 6 months of age and
#: the share of identified infections (0-3 years) that were primary.
VELLORE_FRAC_INFECTED_BY_6M = 0.56
VELLORE_PRIMARY_SHARE = 0.30


def relative_symptomatic_probability(prior_infections: int) -> float:
    """P(diarrhoea | infection) after n previous infections relative to a
    first infection, from Velazquez's two sets of relative risks: 0.37, 0.42,
    0.24 for n = 1, 2, 3. exp(-lambda * n) with lambda ~ 0.5-1.0 spans them,
    the range used for PathogenConfig.symptomatic_reduction_per_prior_infection."""
    if prior_infections <= 0:
        return 1.0
    k = min(prior_infections, 3) - 1
    return VELAZQUEZ_RR_DIARRHOEA[k] / VELAZQUEZ_RR_INFECTION[k]
