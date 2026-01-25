import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      // WebSocketをバックエンドに転送
      '/ws': {
        target: 'http://localhost:8005',
        ws: true,
      }
    }
  }
})
