import { defineConfig, devices } from '@playwright/test'
import path from 'node:path'

const frontendRoot = import.meta.dirname
const repoRoot = path.resolve(frontendRoot, '..')

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: [
    {
      command: 'npm run dev',
      cwd: frontendRoot,
      url: 'http://localhost:5173',
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
    },
    {
      command: 'venv/bin/uvicorn webapp.app:app --host 0.0.0.0 --port 8000',
      cwd: repoRoot,
      url: 'http://localhost:8000/health',
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
      env: {
        WEBAPP_ENV: 'dev',
        WEBAPP_SHARED_PASSWORD: 'e2e-test-password',
      },
    },
  ],
})
