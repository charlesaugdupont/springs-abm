import { test, expect } from '@playwright/test'
import { login, expandAllParamSections } from './helpers'

test('scenario form renders all categories driven by the API, and a run completes', async ({ page, request }) => {
  await login(page)

  // Cross-check against the live API rather than a hardcoded expected
  // count, so this test doesn't need updating if a parameter is added -
  // it fails only if the FORM and the REGISTRY disagree, which is the
  // actual thing worth testing (requirement #4: no hardcoded field lists
  // in the frontend).
  const paramsResp = await request.get('http://localhost:8000/api/parameters', {
    headers: { Cookie: (await page.context().cookies()).map((c) => `${c.name}=${c.value}`).join('; ') },
  })
  const paramsBody = await paramsResp.json()
  const expectedCategories: string[] = paramsBody.by_category.map((c: { category: string }) => c.category)
  const expectedEditableCount = paramsBody.by_category.reduce(
    (sum: number, c: { editable: unknown[] }) => sum + c.editable.length,
    0,
  )

  for (const category of expectedCategories) {
    // Target the accordion trigger specifically by role - several
    // categories share a word with a pathogen toggle label or a field's
    // own label (e.g. "Rotavirus" the category vs. "Base rotavirus
    // severity" the field), so a plain text match is ambiguous.
    await expect(page.getByRole('button', { name: new RegExp(`^${category}`) })).toBeVisible()
  }

  // Sections start collapsed (their fields are unmounted), so expand them all
  // before counting the sliders/inputs they contain.
  await expandAllParamSections(page)

  // Every editable field renders a slider/input - count elements with
  // role="slider" or type=number inputs inside the accordion content.
  const sliderCount = await page.getByRole('slider').count()
  const numberInputCount = await page.locator('input[type="number"]').count()
  expect(sliderCount + numberInputCount).toBe(expectedEditableCount)

  await page.screenshot({ path: 'e2e/screenshots/simulation-form.png', fullPage: true })

  await page.getByRole('button', { name: 'Run simulation' }).click()
  await expect(page.getByText(/Queued…|Running/).first()).toBeVisible({ timeout: 10_000 })

  await expect(page.getByText('Under-5 agents')).toBeVisible({ timeout: 60_000 })
  await page.screenshot({ path: 'e2e/screenshots/simulation-results.png', fullPage: true })
})
