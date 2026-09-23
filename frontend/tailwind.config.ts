import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#17201d",
        paper: "#f4f2eb",
        moss: "#41645a",
        citron: "#d9ef87"
      },
      boxShadow: {
        panel: "0 24px 70px -42px rgba(23, 32, 29, 0.55)"
      }
    }
  },
  plugins: []
} satisfies Config;
