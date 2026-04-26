import { apiClient } from './client';

// ==========================================
// 1. 类型定义 (Types/Interfaces)
// ==========================================

// 用户信息类型
export interface UserInfo {
  id: string | number;
  username: string;
  email?: string;
  avatar?: string;
  // 可以根据你后端的实际返回字段在此处继续补充
}

// 登录接口返回的数据类型
export interface AuthResponse {
  token: string;
  user: UserInfo;
}

// 注册参数类型
export interface RegisterParams {
  username: string;
  password: string;
  email?: string;
}

// ==========================================
// 2. API 封装
// ==========================================

export const userApi = {
  /**
   * 用户登录
   * @param username 用户名
   * @param password 密码
   * @returns 包含 token 和用户信息的对象
   */
  async login(username: string, password: string): Promise<AuthResponse> {
    const { data } = await apiClient.post<AuthResponse>('/api/auth/login', { 
      username, 
      password 
    });
    return data;
  },

  /**
   * 用户注册
   * @param params 注册所需的参数
   */
  async register(params: RegisterParams): Promise<AuthResponse> {
    const { data } = await apiClient.post<AuthResponse>('/api/auth/register', params);
    return data;
  },

  /**
   * 获取当前登录用户信息
   * @returns 当前用户信息
   */
  async profile(): Promise<UserInfo> {
    const { data } = await apiClient.get<UserInfo>('/api/auth/me');
    return data;
  },

  /**
   * 退出登录 (如果后端需要调用接口清除状态)
   */
  async logout(): Promise<void> {
    const { data } = await apiClient.post('/api/auth/logout');
    return data;
  },

  /**
   * 发送重置密码邮件
   * @param email 用户注册邮箱
   */
  async forgotPassword(email: string): Promise<any> {
    const { data } = await apiClient.post('/api/auth/forgot-password', { email });
    return data;
  }
};
