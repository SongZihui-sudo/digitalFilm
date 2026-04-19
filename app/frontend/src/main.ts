import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import { router } from './router'

import './assets/styles/reset.css'
import './assets/styles/darkroom.css'

import { useThemeStore } from './stores/themeStore'
import { ThemeService } from './services/ThemeService'
import { useTheme } from './composables/useTheme'

const app = createApp(App)
const pinia = createPinia()

app.use(pinia)
app.use(router)

const themeService = new ThemeService()
const themeStore = useThemeStore()

const storedTheme = themeService.getStoredTheme()
const initialTheme = storedTheme || themeService.getSystemTheme()
themeStore.setTheme(initialTheme)
themeService.applyTheme(initialTheme)

const { initTheme } = useTheme()
initTheme()

app.mount('#app')
