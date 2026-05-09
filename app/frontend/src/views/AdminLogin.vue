<template>
  <div class="admin-login-shell">
    <div class="admin-login-card panel-card">
      <div class="admin-login-header">
        <div class="brand-dot"></div>
        <h2>DigitalFilm 管理后台</h2>
      </div>

      <div class="form-item">
        <label>管理员账号</label>
        <input
          v-model="username"
          type="text"
          placeholder="请输入管理员账号"
          class="input-base"
          @keyup.enter="handleLogin"
        />
      </div>
      <div class="form-item">
        <label>密码</label>
        <input
          v-model="password"
          type="password"
          placeholder="请输入密码"
          class="input-base"
          @keyup.enter="handleLogin"
        />
      </div>

      <div v-if="errorMessage" class="error-msg">{{ errorMessage }}</div>

      <button class="primary-btn full-width" @click="handleLogin" :disabled="loading || !isValid">
        {{ loading ? '登录中...' : '登录' }}
      </button>

      <div class="admin-login-footer">
        <router-link to="/" class="back-link">← 返回暗房</router-link>
      </div>
    </div>

    <div class="admin-login-theme">
      <ThemeToggle />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useAdminStore } from '@/stores/adminStore';
import ThemeToggle from '@/components/common/ThemeToggle.vue';

const router = useRouter();
const adminStore = useAdminStore();

const username = ref('');
const password = ref('');
const loading = ref(false);
const errorMessage = ref('');

const isValid = computed(() => username.value.trim() && password.value.trim());

async function handleLogin() {
  if (!isValid.value) return;

  loading.value = true;
  errorMessage.value = '';

  try {
    await adminStore.loginAction(username.value, password.value);
    router.push('/admin/dashboard');
  } catch (error: any) {
    errorMessage.value = error.response?.data?.error || '账号或密码错误，或非管理员账号';
  } finally {
    loading.value = false;
  }
}
</script>

<style scoped>
.admin-login-shell {
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 24px;
  background:
    radial-gradient(circle at top left, rgba(124, 140, 255, 0.08), transparent 30%),
    radial-gradient(circle at top right, rgba(48, 196, 141, 0.05), transparent 22%),
    var(--bg-app);
  transition: background 240ms ease;
}

.admin-login-card {
  width: 100%;
  max-width: 400px;
  padding: 32px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.admin-login-header {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 8px;
}

.brand-dot {
  width: 14px;
  height: 14px;
  border-radius: 999px;
  background: linear-gradient(135deg, #7c8cff, #30c48d);
  box-shadow: 0 0 24px rgba(124, 140, 255, 0.45);
  flex-shrink: 0;
}

.admin-login-header h2 {
  margin: 0;
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
}

.form-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form-item label {
  font-size: 14px;
  color: var(--text-secondary);
  font-weight: 500;
}

.error-msg {
  color: #ff3b30;
  font-size: 13px;
  background: rgba(255, 59, 48, 0.1);
  padding: 8px 12px;
  border-radius: 8px;
}

.full-width {
  width: 100%;
}

.admin-login-footer {
  text-align: center;
}

.back-link {
  color: var(--text-muted);
  font-size: 13px;
  text-decoration: none;
  transition: color 160ms ease;
}

.back-link:hover {
  color: var(--accent);
}

.admin-login-theme {
  position: fixed;
  top: 16px;
  right: 16px;
}
</style>
