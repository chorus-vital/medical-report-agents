import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The built bundle is served by FastAPI from src/static/app/ at /; the
// original index.html is still reachable at /legacy.
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
