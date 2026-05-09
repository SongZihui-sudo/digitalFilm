import { apiClient } from './client';

export interface AdminUser {
  id: string;
  username: string;
  email: string;
  isAdmin: boolean;
  createdAt: string;
}

export interface AdminAuthResponse {
  ok: boolean;
  token: string;
  user: AdminUser;
}

export interface AdminListResponse {
  ok: boolean;
  users: AdminUser[];
}

export const adminApi = {
  async login(username: string, password: string): Promise<AdminAuthResponse> {
    const { data } = await apiClient.post<AdminAuthResponse>('/api/admin/login', {
      username,
      password,
    });
    return data;
  },

  async listUsers(): Promise<AdminListResponse> {
    const { data } = await apiClient.get<AdminListResponse>('/api/admin/users');
    return data;
  },

  async createUser(username: string, password: string, email?: string): Promise<{ ok: boolean; user: AdminUser }> {
    const { data } = await apiClient.post<{ ok: boolean; user: AdminUser }>('/api/admin/users', {
      username,
      password,
      email: email || '',
    });
    return data;
  },

  async changePassword(userId: string, newPassword: string): Promise<{ ok: boolean; message: string }> {
    const { data } = await apiClient.put<{ ok: boolean; message: string }>(
      `/api/admin/users/${userId}/password`,
      { new_password: newPassword }
    );
    return data;
  },

  async deleteUser(userId: string): Promise<{ ok: boolean; message: string }> {
    const { data } = await apiClient.delete<{ ok: boolean; message: string }>(
      `/api/admin/users/${userId}`
    );
    return data;
  },

  async toggleAdmin(userId: string): Promise<{ ok: boolean; user: AdminUser }> {
    const { data } = await apiClient.put<{ ok: boolean; user: AdminUser }>(
      `/api/admin/users/${userId}/toggle-admin`
    );
    return data;
  },
};
