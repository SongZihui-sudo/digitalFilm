import { apiClient } from './client';

export const userApi = {
  async login(username: string, password: string) {
    const { data } = await apiClient.post('/api/auth/login', { username, password });
    return data;
  },

  async profile() {
    const { data } = await apiClient.get('/api/auth/me');
    return data;
  },
};
