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
  const userToken = (userStore && (userStore.token?.value ?? userStore.token)) || null
  const adminToken = localStorage.getItem('admin_token')
  // 看当前的 api 是不是需要管理员的
  const isAdminPath = config.url && config.url.startsWith('/api/admin')

  // admin 接口：优先用管理员 token，否则降级到普通 token
  if (isAdminPath) {
    if (adminToken) {
      config.headers['Authorization'] = `Bearer ${adminToken}`
    } else if (userToken) {
      config.headers['Authorization'] = `Bearer ${userToken}`
    }
  } else {
    // 普通接口：优先用用户 token
    if (userToken) {
      config.headers['Authorization'] = `Bearer ${userToken}`
    } else if (adminToken) {
      config.headers['Authorization'] = `Bearer ${adminToken}`
    }
  }
  return config
}

// 挂载拦截器
apiClient.interceptors.request.use(authInterceptor)
static_client.interceptors.request.use(authInterceptor)
image_client.interceptors.request.use(authInterceptor)
