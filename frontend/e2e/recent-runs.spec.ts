import { test, expect } from '@playwright/test'
import { login, expandAllParamSections } from './helpers'

test('recent runs: two submissions create history, params survive re-run, loading a past run repopulates the form, and /runs/:jobId survives a hard refresh', async ({ page }) => {
  await login(page)
  await expect(page.getByText('Population & Demographics')).toBeVisible()

  // The Population size / seed fields live inside collapsed sections by default.
  await expandAllParamSections(page)

  const populationSlider = page.getByRole('slider', { name: 'Population size' })
  const seedInput = page.getByRole('spinbutton', { name: 'Random seed' })

  // First run at the default population size (Radix Slider exposes its
  // current value via aria-valuenow - read that directly rather than
  // scraping the formatted display text next to it).
  const defaultPopulation = await populationSlider.getAttribute('aria-valuenow')
  const seedBefore = await seedInput.inputValue()

  // Other e2e specs share this same real dev server/database (that's the
  // point of Phase 3 - real persistence), so the panel may already have
  // entries from earlier test files. Measure growth relative to that
  // baseline rather than assuming a pristine empty history.
  const baselineViewCount = await page.getByRole('button', { name: 'View' }).count()

  await page.getByRole('button', { name: 'Run simulation' }).click()
  await expect(page.getByText('Under-5 agents')).toBeVisible({ timeout: 60_000 })
  const firstUrl = page.url()
  expect(firstUrl).toMatch(/\/runs\/[a-f0-9]+$/)

  // Tweak the population slider, then confirm an UNTOUCHED field (seed)
  // kept its value across the re-run.
  await populationSlider.focus()
  await populationSlider.press('ArrowRight')
  await populationSlider.press('ArrowRight')
  const tweakedPopulation = await populationSlider.getAttribute('aria-valuenow')
  expect(tweakedPopulation).not.toBe(defaultPopulation)

  await page.getByRole('button', { name: 'Run simulation' }).click()
  await expect(page.getByText('Under-5 agents')).toBeVisible({ timeout: 60_000 })
  const secondUrl = page.url()
  expect(secondUrl).not.toBe(firstUrl)

  const seedAfter = await seedInput.inputValue()
  expect(seedAfter).toBe(seedBefore)

  // Recent runs panel should have exactly 2 more entries than before.
  const viewButtons = page.getByRole('button', { name: 'View' })
  await expect(viewButtons).toHaveCount(baselineViewCount + 2)

  await page.screenshot({ path: 'e2e/screenshots/recent-runs-panel.png', fullPage: true })

  // "Load parameters" on this test's OLDER run - index 1 in the newest-
  // first list (index 0 is the just-submitted tweaked-population run) -
  // repopulates the form with that run's exact, untweaked population value.
  const loadButtons = page.getByRole('button', { name: 'Load parameters' })
  await loadButtons.nth(1).click()
  await expect(populationSlider).toHaveAttribute('aria-valuenow', defaultPopulation!)

  // A hard refresh on /runs/:jobId must still show that run's results.
  await page.reload()
  await expect(page.getByText('Under-5 agents')).toBeVisible({ timeout: 10_000 })
})
