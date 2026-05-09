// src/stores/userStore.ts
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { userApi, type UserInfo, type RegisterParams } from '@/api/userApi'
import { useProjectStore } from '@/stores/projectStore'

export const useUserStore = defineStore('user', () => {
  const token = ref<string | null>(localStorage.getItem('token') || null)
  const userInfo = ref<UserInfo | null>(null)

  const isLoggedIn = computed(() => !!token.value)

  function setToken(newToken: string) {
    token.value = newToken
    localStorage.setItem('token', newToken)
  }

  function setUserInfo(info: any) {
    if (!info) {
      userInfo.value = null
      return
    }

    const normalized = {
      id: info.id ?? info.ID ?? null,
      username: info.username ?? info.Username ?? info.userName ?? '',
      name: info.name ?? info.Name ?? info.fullName ?? info.displayName ?? '',
      email: info.email ?? info.Email ?? '',
      avatar: info.avatar ?? info.Avatar ?? info.avatarUrl ?? ''
    }

    // 保留常见大写/兼容字段，避免模板中的旧写法出错
    ;(normalized as any).Username = normalized.username
    ;(normalized as any).Name = normalized.name
    ;(normalized as any).Email = normalized.email
    ;(normalized as any).Avatar = normalized.avatar

    userInfo.value = normalized as UserInfo
  }

  function logout() {
    token.value = null
    userInfo.value = null
    localStorage.removeItem('token')
    // 在运行时取得 Project Store，避免在模块初始化时调用 Pinia store
    const projectStore = useProjectStore()
    // 统一使用自定义的 reset 方法清空 project store
    projectStore.reset()
  }

  // 1. 登录流程
  async function loginAction(username: string, password: string) {
    try {
      const data = await userApi.login(username, password)
      setToken(data.token)
      setUserInfo(data.user)
      // 在运行时取得 Project Store，避免在模块初始化时调用 Pinia store
      const projectStore = useProjectStore()
      await projectStore.fetchUserProjects();
      return true
    } catch (error) {
      console.error('登录失败:', error)
      throw error
    }
  }

  // 2. 注册流程
  async function registerAction(payload: RegisterParams) {
    try {
      const data = await userApi.register(payload)
      return data
    } catch (error) {
      console.error('注册失败:', error)
      throw error
    }
  }

  // 3. 忘记密码流程
  async function forgotPasswordAction(email: string) {
    try {
      const data = await userApi.forgotPassword(email)
      return data
    } catch (error) {
      console.error('发送重置邮件失败:', error)
      throw error
    }
  }

  // 4. 初始化获取用户信息
  async function fetchProfile() {
    if (!token.value) return
    try {
      console.debug('[userStore] fetchProfile: token=', token.value)
      const resp = await userApi.profile()
      console.debug('[userStore] profile response:', resp)
      // 兼容后端可能返回 { user: {...} } 或直接返回用户对象
      const user = resp && (resp.user ?? resp)
      setUserInfo(user)
    } catch (error) {
      console.error('[userStore] fetchProfile error:', error)
      logout()
    }
  }

  return {
    token,
    userInfo,
    isLoggedIn,
    loginAction,
    registerAction,       // <-- 暴露注册方法
    forgotPasswordAction, // <-- 暴露找回密码方法
    logout,
    fetchProfile
  }
})
