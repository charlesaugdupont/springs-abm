import { test, expect } from '@playwright/test'
import { login } from './helpers'

test('all result charts render after a completed run, and the spatial scrubber visibly changes day-to-day', async ({ page }) => {
  await login(page)
  await expect(page.getByText('Population & Demographics')).toBeVisible()

  await page.getByRole('button', { name: 'Run simulation' }).click()
  await expect(page.getByText('Under-5 agents')).toBeVisible({ timeout: 60_000 })

  const prevalenceCard = page.getByTestId('chart-card-prevalence')
  await expect(prevalenceCard.locator('canvas')).toBeVisible()
  await prevalenceCard.screenshot({ path: 'e2e/screenshots/chart-prevalence.png' })

  const illnessCard = page.getByTestId('chart-card-illness-days')
  await expect(illnessCard.locator('canvas')).toBeVisible()
  await illnessCard.screenshot({ path: 'e2e/screenshots/chart-illness-days.png' })

  const spatialCard = page.getByTestId('chart-card-spatial')
  await spatialCard.scrollIntoViewIfNeeded()
  // The heatmap series legitimately renders on two ZRender canvas layers
  // (base + hover/emphasis, same zrender instance id, different layer id)
  // - not a duplicate mount, so .first() rather than expecting exactly one.
  await expect(spatialCard.locator('canvas').first()).toBeVisible()

  // Day 0 (start of the outbreak) vs. the last day (fullest picture) should
  // look visibly different - proof the scrubber's targeted setOption
  // actually redraws, not just that the slider's own thumb moves.
  const daySlider = page.getByRole('slider', { name: 'Simulation day' })
  await daySlider.focus()
  await daySlider.press('Home')
  // The day label is 1-indexed and shows the total ("Day 1 / 150").
  await expect(page.getByText(/^Day 1 \/ \d+$/)).toBeVisible()
  const dayZeroShot = await spatialCard.screenshot({ path: 'e2e/screenshots/chart-spatial-day0.png' })

  await daySlider.press('End')
  await expect(page.getByText(/^Day \d+ \/ \d+$/)).not.toHaveText(/^Day 1 \//)
  const lastDayShot = await spatialCard.screenshot({ path: 'e2e/screenshots/chart-spatial-lastday.png' })

  expect(Buffer.compare(dayZeroShot, lastDayShot)).not.toBe(0)
})
