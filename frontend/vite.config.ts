import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'node:path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/static': 'http://localhost:8000',
    },
  },
  build: {
    // FastAPI serves this directly (see webapp/app.py) - building straight
    // here avoids a manual copy step in local dev. The Dockerfile's build
    // stage does the equivalent as an explicit COPY, since the two stages
    // don't share a filesystem.
    outDir: '../webapp/frontend_dist',
    emptyOutDir: true,
  },
})
