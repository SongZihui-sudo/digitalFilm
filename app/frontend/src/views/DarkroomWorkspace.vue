<template>
  <div class="workspace-shell">
    <header class="workspace-header">
      <div class="workspace-header__brand">
        <div class="brand-dot"></div>
        <div>
          <div class="brand-title">DigitalFilm Darkroom</div>
          <div class="brand-subtitle">AI-powered photography workflow</div>
        </div>
      </div>

      <div class="workspace-header__actions">
        <button class="theme-toggle" @click="toggleTheme">
          <span class="theme-toggle__label">主题</span>
          <span class="theme-toggle__value">{{ currentThemeLabel }}</span>
        </button>

        <button class="secondary-btn" @click="openFileDialog">
          导入照片
        </button>

        <input
          ref="fileInputRef"
          type="file"
          accept="image/*"
          class="hidden-file-input"
          @change="handleFileChange"
        />

        <button class="primary-btn" @click="handleCreateProject">
          新建项目
        </button>

        <button class="danger-btn" @click="handleDeleteProject">
          删除项目
        </button>

        <LoginModal v-model:visible="showLoginModal" />
      </div>
    </header>

    <div class="workspace-body">
      <LeftSidebar />
      <MainPreview />
      <RightPanel />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useProjectManager } from '@/composables/useProjectManager'
import LeftSidebar from '@/components/layout/LeftSidebar.vue'
import MainPreview from '@/components/layout/MainPreview.vue'
import RightPanel from '@/components/layout/RightPanel.vue'
import { useTheme } from '@/composables/useTheme'
import { useUserStore } from '@/stores/userStore'
import LoginModal from '@/components/common/LoginModal.vue'

const fileInputRef = ref<HTMLInputElement | null>(null)

const { uploadImage, createProject, deleteProject} = useProjectManager()
const { currentTheme, toggleTheme } = useTheme()

const userStore = useUserStore()
const showLoginModal = ref(false)

function openFileDialog() {
  fileInputRef.value?.click()
}

async function handleFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]

  if (!file) return

  // 1. 检查登录状态
  if (!userStore.isLoggedIn) {
    // 如果未登录，弹出登录模态框
    showLoginModal.value = true
    // 重置 input，防止关闭登录框后无法再次触发同一个文件的上传
    input.value = ''
    return
  }

  // 2. 已登录，执行上传
  try {
    await uploadImage(file)
  } catch (error) {
    console.error('上传失败:', error)
  } finally {
    // 允许重复上传同一个文件
    input.value = ''
  }
}

async function handleDeleteProject() {
  // 1. 拦截未登录用户
  if (!userStore.isLoggedIn) {
    showLoginModal.value = true
    return
  }

  try {
    await deleteProject()
  } catch(error) {
    console.error('delete project failed:', error)
    alert('删除项目失败')
  }
}

async function handleCreateProject() {
  // 1. 拦截未登录用户（在弹出输入框之前拦截体验更好）
  if (!userStore.isLoggedIn) {
    showLoginModal.value = true
    return
  }

  const name = window.prompt('请输入新项目名称')
  if (!name) return

  try {
    await createProject(name)
  } catch (error) {
    console.error('create project failed:', error)
    alert('创建项目失败')
  }
}

const currentThemeLabel = computed(() => {
  if (currentTheme.value === 'dark') return '深色'
  if (currentTheme.value === 'light') return '浅色'
  return '跟随系统'
})
</script>

<style scoped>
.workspace-shell {
  width: 100%;
  height: 100vh;
  display: grid;
  grid-template-rows: 72px 1fr;
  background: transparent;
}

.workspace-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 22px;
  border-bottom: 1px solid var(--border-color);
  background: color-mix(in srgb, var(--bg-app) 78%, transparent);
  backdrop-filter: blur(18px);
  transition:
    background 240ms ease,
    border-color 240ms ease;
}

.workspace-header__brand {
  display: flex;
  align-items: center;
  gap: 14px;
}

.brand-dot {
  width: 14px;
  height: 14px;
  border-radius: 999px;
  background: linear-gradient(135deg, #7c8cff, #30c48d);
  box-shadow: 0 0 24px rgba(124, 140, 255, 0.45);
}

.brand-title {
  font-size: 16px;
  font-weight: 700;
  color: var(--text-primary);
}

.brand-subtitle {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 2px;
}

.workspace-header__actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.workspace-body {
  min-height: 0;
  display: grid;
  grid-template-columns: 300px minmax(0, 1fr) 360px;
  gap: 16px;
  padding: 16px;
}

.theme-toggle {
  height: 40px;
  padding: 0 14px;
  border: 1px solid var(--border-color);
  border-radius: 12px;
  background: var(--surface-1);
  color: var(--text-primary);
  display: inline-flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  transition:
    background 160ms ease,
    border-color 160ms ease,
    color 160ms ease,
    transform 160ms ease,
    box-shadow 160ms ease;
}

.theme-toggle:hover {
  background: var(--surface-2);
  border-color: var(--border-strong);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

.theme-toggle__label {
  color: var(--text-secondary);
  font-size: 13px;
}

.theme-toggle__value {
  font-size: 13px;
  font-weight: 600;
}

.hidden-file-input {
  display: none;
}
</style>
