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

  async deleteProject(id: string) {
    const { data } = await apiClient.delete(`/api/projects/${id}`);
    return data;
  }
};
