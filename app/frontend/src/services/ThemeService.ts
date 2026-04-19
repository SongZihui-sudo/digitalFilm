import type { ThemeMode, ResolvedTheme } from '@/stores/themeStore'

const STORAGE_KEY = 'digitalfilm-theme'

export class ThemeService {
  private mediaQuery: MediaQueryList | null = null

  getStoredTheme(): ThemeMode | null {
    const value = localStorage.getItem(STORAGE_KEY)
    if (value === 'light' || value === 'dark' || value === 'system') {
      return value
    }
    return null
  }

  setStoredTheme(theme: ThemeMode) {
    localStorage.setItem(STORAGE_KEY, theme)
  }

  getSystemTheme(): ResolvedTheme {
    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light'
  }

  resolveTheme(theme: ThemeMode): ResolvedTheme {
    if (theme === 'system') {
      return this.getSystemTheme()
    }
    return theme
  }

  applyTheme(resolvedTheme: ResolvedTheme) {
    document.documentElement.setAttribute('data-theme', resolvedTheme)
  }

  watchSystemThemeChange(callback: (theme: ResolvedTheme) => void) {
    this.mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')

    const handler = (e: MediaQueryListEvent) => {
      callback(e.matches ? 'dark' : 'light')
    }

    // 兼容现代浏览器
    if (this.mediaQuery.addEventListener) {
      this.mediaQuery.addEventListener('change', handler)
    } else {
      // 兼容旧写法
      this.mediaQuery.addListener(handler)
    }

    return () => {
      if (!this.mediaQuery) return

      if (this.mediaQuery.removeEventListener) {
        this.mediaQuery.removeEventListener('change', handler)
      } else {
        this.mediaQuery.removeListener(handler)
      }
    }
  }
}
