import type { Page } from '@playwright/test'

export const E2E_PASSWORD = 'e2e-test-password'

export async function login(page: Page, password = E2E_PASSWORD) {
  await page.goto('/login')
  await page.getByLabel('Password').fill(password)
  await page.getByRole('button', { name: 'Log in' }).click()
  await page.waitForURL('/')
}
