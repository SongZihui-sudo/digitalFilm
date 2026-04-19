import { computed } from 'vue'
import {
  useThemeStore,
  type ThemeMode,
  type ResolvedTheme,
} from '@/stores/themeStore'
import { ThemeService } from '@/services/ThemeService'

const themeService = new ThemeService()

let stopWatchSystemTheme: null | (() => void) = null

export function useTheme() {
  const themeStore = useThemeStore()

  const currentTheme = computed(() => themeStore.theme)
  const resolvedTheme = computed(() => themeStore.resolvedTheme)
  const isDark = computed(() => themeStore.resolvedTheme === 'dark')

  function syncResolvedTheme(theme: ThemeMode) {
    const resolved = themeService.resolveTheme(theme)
    themeStore.setTheme(theme)
    themeStore.setResolvedTheme(resolved)
    themeService.applyTheme(resolved)
    themeService.setStoredTheme(theme)
  }

  function applyTheme(theme: ThemeMode) {
    syncResolvedTheme(theme)

    if (stopWatchSystemTheme) {
      stopWatchSystemTheme()
      stopWatchSystemTheme = null
    }

    if (theme === 'system') {
      stopWatchSystemTheme = themeService.watchSystemThemeChange(
        (newResolvedTheme: ResolvedTheme) => {
          themeStore.setResolvedTheme(newResolvedTheme)
          themeService.applyTheme(newResolvedTheme)
        }
      )
    }
  }

  function toggleTheme() {
    const current = themeStore.theme

    let nextTheme: ThemeMode

    if (current === 'dark') {
      nextTheme = 'light'
    } else if (current === 'light') {
      nextTheme = 'system'
    } else {
      nextTheme = 'dark'
    }

    applyTheme(nextTheme)
  }

  function initTheme() {
    const stored = themeService.getStoredTheme()
    const initialTheme: ThemeMode = stored || 'system'
    applyTheme(initialTheme)
  }

  return {
    currentTheme,
    resolvedTheme,
    isDark,
    applyTheme,
    toggleTheme,
    initTheme,
  }
}
