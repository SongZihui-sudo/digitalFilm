import { apiClient } from './client';

export const projectApi = {
  async listProjects() {
    const { data } = await apiClient.get('/api/projects');
    return data;
  },

  async createProject(name: string) {
    const { data } = await apiClient.post('/api/create_projects', { name });
    return data;
  },
};
