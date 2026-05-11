import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // dev-time proxy: all /api/* → FastAPI on :8000
      "/ingest":   { target: "http://localhost:8000", changeOrigin: true },
      "/classify": { target: "http://localhost:8000", changeOrigin: true },
      "/query/stream": { target: "http://localhost:8000", changeOrigin: true },
      "/query":    { target: "http://localhost:8000", changeOrigin: true },
      "/eval":     { target: "http://localhost:8000", changeOrigin: true },
      "/audit":    { target: "http://localhost:8000", changeOrigin: true },
      "/metrics":  { target: "http://localhost:8000", changeOrigin: true },
      "/health":   { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  define: {
    // fallback — override with VITE_API_BASE in .env.production
    "import.meta.env.VITE_API_BASE": JSON.stringify(""),
  },
});
