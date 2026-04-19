import { defineStore } from 'pinia'

export type ThemeMode = 'light' | 'dark' | 'system'
export type ResolvedTheme = 'light' | 'dark'

interface ThemeState {
  theme: ThemeMode
  resolvedTheme: ResolvedTheme
}

export const useThemeStore = defineStore('theme', {
  state: (): ThemeState => ({
    theme: 'system',
    resolvedTheme: 'dark',
  }),

  actions: {
    setTheme(theme: ThemeMode) {
      this.theme = theme
    },

    setResolvedTheme(theme: ResolvedTheme) {
      this.resolvedTheme = theme
    },
  },
})
