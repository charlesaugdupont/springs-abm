import type { Page } from '@playwright/test'

export const E2E_PASSWORD = 'e2e-test-password'

export async function login(page: Page, password = E2E_PASSWORD) {
  await page.goto('/login')
  await page.getByLabel('Password').fill(password)
  await page.getByRole('button', { name: 'Log in' }).click()
  await page.waitForURL('/')
}

// The parameter-category accordion now starts fully COLLAPSED by default
// (Radix unmounts closed content), so any test that needs to read or interact
// with a field (slider, number input, info tooltip) must open the sections
// first. Clicks every accordion trigger that isn't already expanded.
export async function expandAllParamSections(page: Page) {
  const triggers = page.locator('[data-slot="accordion-trigger"]')
  const count = await triggers.count()
  for (let i = 0; i < count; i++) {
    const trigger = triggers.nth(i)
    if ((await trigger.getAttribute('aria-expanded')) !== 'true') {
      await trigger.click()
    }
  }
}
