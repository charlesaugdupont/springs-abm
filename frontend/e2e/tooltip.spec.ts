import { test, expect } from '@playwright/test'
import { login, expandAllParamSections } from './helpers'

// Named regression test for the specific bug the plan traced: the old
// tooltip was clipped by an `overflow:hidden` ancestor and had zero
// viewport-collision handling, so it visibly overflowed when opened on a
// field in the right-hand column of the 2-column parameter grid. The
// Radix-based shadcn Tooltip (see src/components/ui/tooltip.tsx) portals
// its content out of any clipping ancestor and includes built-in collision
// detection - this test proves that in practice, not just in theory.
test('info tooltip on a right-column field stays fully inside the viewport', async ({ page }) => {
  await login(page)

  const viewport = page.viewportSize()
  expect(viewport).not.toBeNull()

  // Wait for the registry-driven form to actually finish loading (a
  // network round trip) before querying for fields - otherwise this races
  // the initial fetch and finds nothing.
  await expect(page.getByText('Population & Demographics')).toBeVisible()

  // Fields (and their info buttons) are inside collapsed sections by default.
  await expandAllParamSections(page)

  const infoButtons = page.locator('button[aria-label^="More info about"]')
  const count = await infoButtons.count()
  expect(count).toBeGreaterThan(0)

  // Find one sitting in the right half of the viewport - the exact
  // position where the old bug manifested.
  let target = null
  for (let i = 0; i < count; i++) {
    const box = await infoButtons.nth(i).boundingBox()
    if (box && box.x > viewport!.width / 2) {
      target = infoButtons.nth(i)
      break
    }
  }
  expect(target, 'expected at least one info button in the right-hand column').not.toBeNull()

  // Radix Tooltip opens on hover/focus, not click.
  await target!.hover()
  const tooltip = page.locator('[data-slot="tooltip-content"]')
  await expect(tooltip).toBeVisible()

  await page.screenshot({ path: 'e2e/screenshots/tooltip-right-column.png' })

  const tooltipBox = await tooltip.boundingBox()
  expect(tooltipBox).not.toBeNull()
  expect(tooltipBox!.x).toBeGreaterThanOrEqual(0)
  expect(tooltipBox!.y).toBeGreaterThanOrEqual(0)
  expect(tooltipBox!.x + tooltipBox!.width).toBeLessThanOrEqual(viewport!.width)
  expect(tooltipBox!.y + tooltipBox!.height).toBeLessThanOrEqual(viewport!.height)
})
