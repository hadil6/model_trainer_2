import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on("proxyReq", (proxyReq, req) => {
            // SSE streams must not be gzip-compressed — compressed streams buffer
            // until the chunk fills and never flush individual events in real time.
            if (req.url?.includes("/events")) {
              proxyReq.removeHeader("accept-encoding");
            }
          });
          proxy.on("proxyRes", (_proxyRes, req, res) => {
            if (req.url?.includes("/events")) {
              res.setHeader("X-Accel-Buffering", "no");
              res.setHeader("Cache-Control", "no-cache");
            }
          });
        },
      },
    },
  },
});
