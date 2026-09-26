import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The built bundle is served by FastAPI from src/static/app/, alongside the
// original index.html — the old UI keeps working until this one is at parity.
export default defineConfig({
  plugins: [react()],
  base: "/static/app/",
  build: {
    outDir: "../src/static/app",
    emptyOutDir: true,
  },
  server: {
    // `npm run dev` talks to the FastAPI server for /api.
    proxy: { "/api": "http://localhost:8000" },
  },
});
