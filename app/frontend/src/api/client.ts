import axios from 'axios';
import { useUserStore } from '@/stores/userStore'

export const apiClient = axios.create({
  baseURL: 'http://127.0.0.1:8080',
  timeout: 30000,
});

export const static_client = axios.create({
  baseURL: 'http://127.0.0.1:6060',
  timeout: 30000,
});

export const image_client = axios.create({
  baseURL: 'http://127.0.0.1:7070',
  timeout: 30000,
});

// 创建一个拦截器
const authInterceptor = (config: any) => {
  const userStore = useUserStore()
  const token = (userStore && (userStore.token?.value ?? userStore.token)) || null
  if (token) {
    config.headers['Authorization'] = `Bearer ${token}`
  } else {
    // 如果没有普通用户 token，尝试使用管理员 token
    const adminToken = localStorage.getItem('admin_token')
    if (adminToken) {
      config.headers['Authorization'] = `Bearer ${adminToken}`
    }
  }
  return config
}

// 挂载拦截器
apiClient.interceptors.request.use(authInterceptor)
static_client.interceptors.request.use(authInterceptor)
image_client.interceptors.request.use(authInterceptor)
