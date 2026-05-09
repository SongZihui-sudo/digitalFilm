<template>
  <Teleport to="body">
    <Transition name="fade">
      <div v-if="visible" class="modal-overlay" @click.self="closeModal">
        <div class="modal-content">
          <div class="modal-header">
            <h3>{{ modalTitle }}</h3>
            <button class="close-btn" @click="closeModal">✕</button>
          </div>

          <div class="modal-body">
            <template v-if="mode === 'login'">
              <div class="form-item">
                <label>用户名</label>
                <input 
                  v-model="loginForm.username" 
                  type="text" 
                  placeholder="请输入用户名" 
                  class="base-input"
                />
              </div>
              <div class="form-item">
                <label>密码</label>
                <input 
                  v-model="loginForm.password" 
                  type="password" 
                  placeholder="请输入密码" 
                  class="base-input"
                  @keyup.enter="handleLogin"
                />
              </div>
            </template>

            <template v-if="mode === 'register'">
              <div class="form-item">
                <label>用户名</label>
                <input 
                  v-model="registerForm.username" 
                  type="text" 
                  placeholder="请输入用户名 (至少3个字符)" 
                  class="base-input"
                />
              </div>
              <div class="form-item">
                <label>邮箱 (可选)</label>
                <input 
                  v-model="registerForm.email" 
                  type="email" 
                  placeholder="请输入邮箱" 
                  class="base-input"
                />
              </div>
              <div class="form-item">
                <label>密码</label>
                <input 
                  v-model="registerForm.password" 
                  type="password" 
                  placeholder="请输入密码 (至少6位)" 
                  class="base-input"
                />
              </div>
              <div class="form-item">
                <label>确认密码</label>
                <input 
                  v-model="registerForm.confirmPassword" 
                  type="password" 
                  placeholder="请再次输入密码" 
                  class="base-input"
                  @keyup.enter="handleRegister"
                />
              </div>
            </template>

            <template v-if="mode === 'forgot'">
              <div class="form-text">
                请输入您的注册邮箱，我们将向您发送重置密码的指引。
              </div>
              <div class="form-item">
                <label>邮箱</label>
                <input 
                  v-model="forgotForm.email" 
                  type="email" 
                  placeholder="请输入注册时使用的邮箱" 
                  class="base-input"
                  @keyup.enter="handleForgot"
                />
              </div>
            </template>

            <div v-if="errorMessage" class="error-msg">{{ errorMessage }}</div>
            <div v-if="successMessage" class="success-msg">{{ successMessage }}</div>
          </div>

          <div class="modal-footer">
            <button 
              class="primary-btn full-width" 
              @click="submitCurrentForm" 
              :disabled="loading || !isCurrentValid"
            >
              {{ submitButtonText }}
            </button>

            <div class="modal-links">
              <template v-if="mode === 'login'">
                <span class="text-link" @click="switchMode('register')">没有账号？去注册</span>
                <span class="text-link" @click="switchMode('forgot')">忘记密码？</span>
              </template>
              <template v-else>
                <span class="text-link" @click="switchMode('login')">返回登录</span>
              </template>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import { useUserStore } from '@/stores/userStore'

const props = defineProps<{ visible: boolean }>()
const emit = defineEmits<{ (e: 'update:visible', value: boolean): void }>()

const userStore = useUserStore()

// 模态框模式：login | register | forgot
type ModalMode = 'login' | 'register' | 'forgot'
const mode = ref<ModalMode>('login')

const loading = ref(false)
const errorMessage = ref('')
const successMessage = ref('')

// ==========================================
// 表单数据定义
// ==========================================
const loginForm = reactive({
  username: '',
  password: ''
})

const registerForm = reactive({
  username: '',
  email: '',
  password: '',
  confirmPassword: ''
})

const forgotForm = reactive({
  email: ''
})

// ==========================================
// 计算属性 (动态标题、按钮文字、校验)
// ==========================================
const modalTitle = computed(() => {
  if (mode.value === 'login') return '登录'
  if (mode.value === 'register') return '注册新账号'
  return '找回密码'
})

const submitButtonText = computed(() => {
  if (loading.value) return '处理中...'
  if (mode.value === 'login') return '登录'
  if (mode.value === 'register') return '立即注册'
  return '发送重置邮件'
})

const isCurrentValid = computed(() => {
  if (mode.value === 'login') {
    return loginForm.username.trim() && loginForm.password.trim()
  }
  if (mode.value === 'register') {
    return (
      registerForm.username.trim().length >= 3 &&
      registerForm.password.trim().length >= 6 &&
      registerForm.password === registerForm.confirmPassword
    )
  }
  if (mode.value === 'forgot') {
    return forgotForm.email.trim().includes('@')
  }
  return false
})

// ==========================================
// 方法与业务逻辑
// ==========================================
function switchMode(newMode: ModalMode) {
  mode.value = newMode
  errorMessage.value = ''
  successMessage.value = ''
}

function closeModal() {
  emit('update:visible', false)
  // 延迟清空表单，避免退场动画时看到内容突变
  setTimeout(() => {
    mode.value = 'login'
    loginForm.username = ''
    loginForm.password = ''
    registerForm.username = ''
    registerForm.email = ''
    registerForm.password = ''
    registerForm.confirmPassword = ''
    forgotForm.email = ''
    errorMessage.value = ''
    successMessage.value = ''
  }, 300)
}

function submitCurrentForm() {
  if (mode.value === 'login') handleLogin()
  else if (mode.value === 'register') handleRegister()
  else if (mode.value === 'forgot') handleForgot()
}

// 处理登录
async function handleLogin() {
  if (!isCurrentValid.value) return
  loading.value = true
  errorMessage.value = ''

  try {
    await userStore.loginAction(loginForm.username, loginForm.password)
    closeModal()
  } catch (error: any) {
    errorMessage.value = error.response?.data?.error || '账号或密码错误'
  } finally {
    loading.value = false
  }
}

// 处理注册
async function handleRegister() {
  if (!isCurrentValid.value) return
  if (registerForm.password !== registerForm.confirmPassword) {
    errorMessage.value = '两次输入的密码不一致'
    return
  }

  loading.value = true
  errorMessage.value = ''
  successMessage.value = ''

  try {
    // 真正调用 Store 中的 registerAction
    await userStore.registerAction({
      username: registerForm.username,
      email: registerForm.email,
      password: registerForm.password
    })
    
    // 注册成功后，直接提示并切换到登录模式，顺便把刚注册的用户名填过去
    successMessage.value = '注册成功！请登录'
    loginForm.username = registerForm.username
    setTimeout(() => {
      switchMode('login')
    }, 1500)
  } catch (error: any) {
    // 根据后端的错误结构提取信息
    errorMessage.value = error.response?.data?.error || '注册失败，用户名可能已被占用'
  } finally {
    loading.value = false
  }
}

// 处理忘记密码
async function handleForgot() {
  if (!isCurrentValid.value) return
  
  loading.value = true
  errorMessage.value = ''
  successMessage.value = ''

  try {
    // 真正调用 Store 中的 forgotPasswordAction
    await userStore.forgotPasswordAction(forgotForm.email)
    
    successMessage.value = '重置指引已发送至您的邮箱，请注意查收。'
    forgotForm.email = ''
  } catch (error: any) {
    errorMessage.value = error.response?.data?.error || '发送失败，请检查邮箱是否正确'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(4px);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 9999;
}

.modal-content {
  width: 100%;
  max-width: 380px;
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.modal-header h3 {
  margin: 0;
  font-size: 20px;
  color: #111;
}

.close-btn {
  background: transparent;
  border: none;
  font-size: 20px;
  color: #999;
  cursor: pointer;
  transition: color 0.2s;
}
.close-btn:hover {
  color: #333;
}

.modal-body {
  display: flex;
  flex-direction: column;
}

.form-text {
  font-size: 13px;
  color: #666;
  line-height: 1.5;
  margin-bottom: 16px;
}

.form-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 16px;
}

.form-item label {
  font-size: 14px;
  color: #555;
  font-weight: 500;
}

.base-input {
  padding: 12px 14px;
  border: 1px solid #ddd;
  border-radius: 8px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.base-input:focus {
  border-color: #007aff;
  box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.1);
}

.error-msg {
  color: #ff3b30;
  font-size: 13px;
  margin-top: -4px;
  margin-bottom: 8px;
  background: rgba(255, 59, 48, 0.1);
  padding: 8px 12px;
  border-radius: 6px;
}

.success-msg {
  color: #34c759;
  font-size: 13px;
  margin-top: -4px;
  margin-bottom: 8px;
  background: rgba(52, 199, 89, 0.1);
  padding: 8px 12px;
  border-radius: 6px;
}

.modal-footer {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.full-width {
  width: 100%;
  padding: 12px;
  font-size: 15px;
  font-weight: 500;
}

.modal-links {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
}

.text-link {
  color: #007aff;
  cursor: pointer;
  transition: opacity 0.2s;
}

.text-link:hover {
  opacity: 0.8;
  text-decoration: underline;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
