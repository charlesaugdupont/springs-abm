import { test, expect } from '@playwright/test'

test('unauthenticated visitors are redirected to login', async ({ page }) => {
  await page.goto('/')
  await page.waitForURL('/login')
  await expect(page.getByText('SPRINGS ABM')).toBeVisible()
  await page.screenshot({ path: 'e2e/screenshots/smoke.png' })
})
