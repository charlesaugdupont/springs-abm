import { test, expect } from '@playwright/test'
import { E2E_PASSWORD } from './helpers'

// Verifies the app served from a SINGLE real uvicorn origin (built
// webapp/frontend_dist/ + /api on :8000), not the split Vite-dev +
// uvicorn proxy setup every other spec exercises via baseURL (:5173).
// Both point at the same Playwright-managed uvicorn instance (see
// playwright.config.ts's webServer array) - this one just navigates the
// browser directly to :8000 instead of going through the Vite dev
// server/proxy, so it needs `npm run build` to have populated
// webapp/frontend_dist/ first (the other specs don't, since dev mode
// serves frontend/src/ directly).
test('single-origin production-style serving: SPA loads, login works, client-side route survives direct navigation', async ({ page }) => {
  await page.goto('http://localhost:8000/login')
  await expect(page.getByText('SPRINGS ABM')).toBeVisible()

  await page.getByLabel('Password').fill(E2E_PASSWORD)
  await page.getByRole('button', { name: 'Log in' }).click()
  await page.waitForURL('http://localhost:8000/')
  await expect(page.getByText('Population & Demographics')).toBeVisible()
  await page.screenshot({ path: 'e2e/screenshots/prod-serving.png', fullPage: true })

  // Direct navigation (not client-side routing) to a nested path - proves
  // the SPA-fallback catch-all in webapp/app.py actually works, not just
  // that client-side <Link> navigation works.
  await page.goto('http://localhost:8000/about')
  await expect(page.getByRole('heading', { name: 'About the model' })).toBeVisible()
})
