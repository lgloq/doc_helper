import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const devPort = Number(process.env.FRONTEND_PORT ?? process.env.PORT ?? 18073);

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: Number.isFinite(devPort) ? devPort : 18073,
    proxy: {
      "/api": {
        target: "http://backend:8000",
        changeOrigin: true,
      },
    },
  },
});
