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
  }
  return config
}

// 挂载拦截器
apiClient.interceptors.request.use(authInterceptor)
static_client.interceptors.request.use(authInterceptor)
image_client.interceptors.request.use(authInterceptor)
