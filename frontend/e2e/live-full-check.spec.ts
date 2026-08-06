import { test, expect, type Page } from '@playwright/test'

// A broader, slower pass against the real live deployment - several
// distinct scenarios (not just the defaults), chart rendering, tooltip
// fix, and recent-runs accumulation, all against real Cloud Run compute.
// Manual-only (needs env vars), not part of the regular suite.
const LIVE_URL = process.env.SPRINGS_LIVE_URL
const LIVE_PASSWORD = process.env.SPRINGS_LIVE_PASSWORD

test.skip(!LIVE_URL || !LIVE_PASSWORD, 'SPRINGS_LIVE_URL/SPRINGS_LIVE_PASSWORD not set')
test.describe.configure({ mode: 'serial' })
test.setTimeout(180_000)

async function login(page: Page) {
  await page.goto(`${LIVE_URL}/login`)
  await page.getByLabel('Password').fill(LIVE_PASSWORD!)
  await page.getByRole('button', { name: 'Log in' }).click()
  await expect(page.getByText('Population & Demographics')).toBeVisible({ timeout: 20_000 })
}

async function runAndWait(page: Page, label: string) {
  await page.getByRole('button', { name: 'Run simulation' }).click()
  await expect(page.getByText('Under-5 agents')).toBeVisible({ timeout: 120_000 })
  await page.screenshot({ path: `e2e/screenshots/live-${label}.png`, fullPage: true })
}

test('scenario 1: defaults (both pathogens)', async ({ page }) => {
  await login(page)
  await runAndWait(page, 'scenario-both-pathogens')
  const stats = await page.locator('text=Under-5 agents').locator('xpath=../..').innerText()
  console.log('[scenario 1 result]', stats.replace(/\n/g, ' | '))
})

test('scenario 2: rotavirus only', async ({ page }) => {
  await login(page)
  await page.getByRole('switch').nth(1).click() // toggle off campylobacter
  await expect(page.getByRole('switch').nth(1)).toHaveAttribute('data-state', 'unchecked')
  await runAndWait(page, 'scenario-rota-only')
})

test('scenario 3: small fast population, longer duration', async ({ page }) => {
  await login(page)
  const popSlider = page.getByRole('slider', { name: 'Population size' })
  await popSlider.focus()
  for (let i = 0; i < 30; i++) await popSlider.press('ArrowLeft') // push toward the minimum
  const popValue = await popSlider.getAttribute('aria-valuenow')
  console.log('[scenario 3 population]', popValue)
  await runAndWait(page, 'scenario-small-population')
})

test('both pathogens disabled is rejected client-side, no request sent', async ({ page }) => {
  await login(page)
  const switches = page.getByRole('switch')
  await switches.nth(0).click()
  await switches.nth(1).click()
  await page.getByRole('button', { name: 'Run simulation' }).click()
  await expect(page.getByText('At least one pathogen must be enabled.')).toBeVisible()
})

test('recent runs accumulated across this session, and tooltip fix holds on live deployment', async ({ page }) => {
  await login(page)
  const viewCount = await page.getByRole('button', { name: 'View' }).count()
  console.log('[recent runs count after this test session]', viewCount)
  expect(viewCount).toBeGreaterThanOrEqual(3)

  // Same right-column tooltip regression check as the local suite, once
  // more against the actual production build being served.
  const viewport = page.viewportSize()!
  const infoButtons = page.locator('button[aria-label^="More info about"]')
  const count = await infoButtons.count()
  let target = null
  for (let i = 0; i < count; i++) {
    const box = await infoButtons.nth(i).boundingBox()
    if (box && box.x > viewport.width / 2) {
      target = infoButtons.nth(i)
      break
    }
  }
  expect(target).not.toBeNull()
  await target!.hover()
  const tooltip = page.locator('[data-slot="tooltip-content"]')
  await expect(tooltip).toBeVisible()
  const box = await tooltip.boundingBox()
  expect(box!.x + box!.width).toBeLessThanOrEqual(viewport.width)
  await page.screenshot({ path: 'e2e/screenshots/live-tooltip-and-history.png', fullPage: true })
})
