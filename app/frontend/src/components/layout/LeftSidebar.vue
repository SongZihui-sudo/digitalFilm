<template>
  <aside class="left-sidebar panel-card">
    <div class="left-sidebar__top">
      <div class="section-title">Library</div>

      <button class="secondary-btn left-sidebar__upload" @click="openFileDialog">
        上传
      </button>

      <input
        ref="fileInputRef"
        type="file"
        accept="image/*"
        class="hidden-file-input"
        @change="handleFileChange"
      />
    </div>

    <div class="left-sidebar__section left-sidebar__section--card">
      <ProjectList />
    </div>

    <div class="left-sidebar__section left-sidebar__section--card left-sidebar__section--grow">
      <ImageThumbnailList />
    </div>

    <div v-if="isLoggedIn && userInfo" class="left-sidebar__user" ref="userWidgetRef">
      <div class="user-trigger" @click="toggleUserInfo">
        <img :src="avatarSrc" alt="Avatar" class="user-avatar" />
        <span class="user-name">{{ userInfo.Username }}</span>
      </div>

      <Transition name="fade-slide">
        <div v-if="showUserInfo" class="user-popover">
          <div class="popover-header">
            <img :src="avatarSrc" alt="Avatar" class="popover-avatar-large" />
            <div class="popover-info">
              <span class="popover-name">{{ userInfo.Username }}</span>
              <span class="popover-email">{{ userInfo.Email }}</span>
            </div>
          </div>
          <div class="popover-actions">
            <button class="secondary-btn full-width" @click="handleLogout">
              退出登录
            </button>
          </div>
        </div>
      </Transition>
    </div>

    <div v-else class="left-sidebar__login">
      <button class="primary-btn full-width" @click="handleLogin">
        登录 / 注册
      </button>
      <LoginModal v-model:visible="showLoginModal" />
    </div>
  </aside>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { storeToRefs } from 'pinia'
import ProjectList from '@/components/project/ProjectList.vue'
import ImageThumbnailList from '@/components/project/ImageThumbnailList.vue'
import { useProjectManager } from '@/composables/useProjectManager'
import { useUserStore } from '@/stores/userStore'
import LoginModal from '@/components/common/LoginModal.vue'
import { generateAvatar } from '@/composables/useAvatar'

const fileInputRef = ref<HTMLInputElement | null>(null)
const { uploadImage } = useProjectManager()
const showLoginModal = ref(false)

// ==========================================
// 引入 User Store
// ==========================================
const userStore = useUserStore()
const { userInfo, isLoggedIn } = storeToRefs(userStore)

const avatarSrc = computed(() => {
  const info = (userInfo as any).value
  if (!info) return ''
  return info.avatar || generateAvatar(info.username || info.name || info.Username || 'U', 64)
})

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

// ==========================================
// 用户信息与登录控制逻辑
// ==========================================
const userWidgetRef = ref<HTMLElement | null>(null)
const showUserInfo = ref(false)

function toggleUserInfo() {
  showUserInfo.value = !showUserInfo.value
}

function handleLogout() {
  showUserInfo.value = false
  showLoginModal.value = false
  userStore.logout()
}

function handleLogin() {
  showLoginModal.value = true
}

// 点击外部关闭弹窗逻辑
function handleClickOutside(event: MouseEvent) {
  if (
    showUserInfo.value && 
    userWidgetRef.value && 
    !userWidgetRef.value.contains(event.target as Node)
  ) {
    showUserInfo.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<style scoped>
.left-sidebar {
  min-height: 0;
  display: flex;
  flex-direction: column;
  padding: 16px;
  overflow: hidden;
}

.left-sidebar__top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.left-sidebar__upload {
  width: 86px;
}

.left-sidebar__section {
  margin-bottom: 12px;
  min-height: 0;
}

.left-sidebar__section--grow {
  flex: 1;
  overflow: hidden;
}

.left-sidebar__section--card {
  padding: 12px;
  border-radius: 16px;
  background: var(--surface-1, #fff);
  border: 1px solid var(--border-color, #eee);
  transition:
    background var(--transition-fast, 0.2s),
    border-color var(--transition-fast, 0.2s);
}

.left-sidebar__section--card:hover {
  background: var(--surface-2, #f9f9f9);
  border-color: var(--border-strong, #ccc);
}

.hidden-file-input {
  display: none;
}

/* ==========================================
   底部用户区域通用样式
========================================== */
.left-sidebar__user,
.left-sidebar__login {
  position: relative;
  padding: 10px 12px;
  border-radius: 16px;
  background: var(--surface-1, #fff);
  border: 1px solid var(--border-color, #eee);
  transition: background 0.2s, border-color 0.2s;
  user-select: none;
}

.left-sidebar__user:hover {
  background: var(--surface-2, #f5f5f5);
}

/* 登录容器特殊处理，让按钮撑满 */
.left-sidebar__login {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px; /* 登录按钮区域稍微收紧一点，视UI而定 */
}

.user-trigger {
  display: flex;
  align-items: center;
  gap: 12px;
  cursor: pointer;
}

.user-avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  object-fit: cover;
  flex-shrink: 0;
}

.user-name {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary, #333);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 悬浮详细信息弹窗 */
.user-popover {
  position: absolute;
  bottom: calc(100% + 10px); 
  left: 0;
  width: 100%;
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
  border: 1px solid #eaeaea;
  padding: 16px;
  z-index: 100;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.popover-header {
  display: flex;
  align-items: center;
  gap: 12px;
}

.popover-avatar-large {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  object-fit: cover;
}

.popover-info {
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.popover-name {
  font-size: 15px;
  font-weight: 600;
  color: #111;
}

.popover-email {
  font-size: 12px;
  color: #888;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.popover-actions {
  border-top: 1px solid #f0f0f0;
  padding-top: 12px;
}

.full-width {
  width: 100%;
  justify-content: center;
}

/* 弹窗过渡动画 */
.fade-slide-enter-active,
.fade-slide-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}
.fade-slide-enter-from,
.fade-slide-leave-to {
  opacity: 0;
  transform: translateY(8px);
}
</style>
