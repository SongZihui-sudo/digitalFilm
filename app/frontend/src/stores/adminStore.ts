import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import { adminApi, type AdminUser } from '@/api/adminApi';

export const useAdminStore = defineStore('admin', () => {
  const token = ref<string | null>(localStorage.getItem('admin_token') || null);
  const adminInfo = ref<AdminUser | null>(null);

  const isLoggedIn = computed(() => !!token.value);

  function setToken(newToken: string) {
    token.value = newToken;
    localStorage.setItem('admin_token', newToken);
  }

  function logout() {
    token.value = null;
    adminInfo.value = null;
    localStorage.removeItem('admin_token');
  }

  async function loginAction(username: string, password: string) {
    const data = await adminApi.login(username, password);
    setToken(data.token);
    adminInfo.value = data.user;
    return true;
  }

  return {
    token,
    adminInfo,
    isLoggedIn,
    loginAction,
    logout,
  };
});
