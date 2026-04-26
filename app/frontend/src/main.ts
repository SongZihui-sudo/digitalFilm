import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import { router } from './router'

import './assets/styles/reset.css'
import './assets/styles/darkroom.css'

import { useThemeStore } from './stores/themeStore'
import { ThemeService } from './services/ThemeService'
import { useTheme } from './composables/useTheme'
import { useUserStore } from '@/stores/userStore'
import { useProjectStore } from '@/stores/projectStore'

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

// 应用启动时，优先完成用户恢复与项目加载，再挂载应用，避免刷新时组件先于用户数据渲染。
const userStore = useUserStore()
const projectStore = useProjectStore()

;(async function initApp() {
	// 兼容 token 为 ref 或普通 string 的情况
	const tokenValue = (userStore.token && typeof (userStore.token as any) === 'object' && 'value' in (userStore.token as any))
		? (userStore.token as any).value
		: (userStore.token as any)

	console.debug('[initApp] token from store:', tokenValue)

	if (tokenValue) {
		try {
			await userStore.fetchProfile()
			await projectStore.fetchUserProjects()
		} catch (e) {
			console.error('Failed to initialize auth or projects:', e)
			// 如果恢复失败，确保用户状态被清理
			userStore.logout()
		}
	}

	app.mount('#app')
})()
