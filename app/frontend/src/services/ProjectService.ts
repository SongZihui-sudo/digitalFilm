import { projectApi } from '@/api/projectApi';

export class ProjectService {
  async listProjects() {
    return await projectApi.listProjects();
  }

  async createProject(name: string) {
    return await projectApi.createProject(name);
  }
}
