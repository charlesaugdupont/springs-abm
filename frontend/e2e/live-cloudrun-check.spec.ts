import { test, expect } from '@playwright/test'

// Manual-run-only check against a real deployed URL - not part of the
// regular suite (needs a live service and a password only known at deploy
// time). Useful after any future redeploy. Run directly:
//   SPRINGS_LIVE_URL=https://your-service-url \
//   SPRINGS_LIVE_PASSWORD=your-password \
//   npx playwright test live-cloudrun-check.spec.ts
const LIVE_URL = process.env.SPRINGS_LIVE_URL
const LIVE_PASSWORD = process.env.SPRINGS_LIVE_PASSWORD

test.skip(!LIVE_URL || !LIVE_PASSWORD, 'SPRINGS_LIVE_URL/SPRINGS_LIVE_PASSWORD not set')

test('live deployment loads and accepts login', async ({ page }) => {
  await page.goto(`${LIVE_URL}/login`)
  await expect(page.getByText('SPRINGS ABM')).toBeVisible()

  await page.getByLabel('Password').fill(LIVE_PASSWORD!)
  await page.getByRole('button', { name: 'Log in' }).click()

  // A generous timeout here on purpose: Cloud Run scale-to-zero means the
  // first hit after any idle period is a genuine cold start (container
  // boot + Python/torch import), not just a slow network round trip.
  await expect(page.getByText('Population & Demographics')).toBeVisible({ timeout: 20_000 })
  await page.screenshot({ path: 'e2e/screenshots/live-deployment.png', fullPage: true })
})
