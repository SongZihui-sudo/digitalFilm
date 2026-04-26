import { defineStore } from 'pinia';
import type { Project } from '@/models/project';
import type { ImageAsset } from '@/models/image';
import { useUserStore } from './userStore'; // 引入 UserStore
import { projectApi } from '@/api/projectApi'; // 假设你有对应的接口文件

interface ProjectState {
  projects: Project[];
  currentProject: Project | null;
  currentImages: ImageAsset[];
  currentImage: ImageAsset | null;
}

export const useProjectStore = defineStore('project', {
  state: (): ProjectState => ({
    projects: [],
    currentProject: null,
    currentImages: [],
    currentImage: null,
  }),

  actions: {
    // 1. 初始化方法：通常在登录成功后调用
    async fetchUserProjects() {
      const userStore = useUserStore();
      if (!userStore.isLoggedIn) return;

      try {
        // 这里调用你之前的加载项目接口
        const data = await projectApi.listProjects();
        this.setProjects(data);
        
        // 默认选中第一个项目（可选）
        if (this.projects.length > 0 && !this.currentProject) {
          this.setCurrentProject(this.projects[0]);
        }
      } catch (error) {
        console.error('Failed to load projects:', error);
      }
    },

    // 2. 重置方法：在注销时调用
    reset() {
      this.projects = [];
      this.currentProject = null;
      this.currentImages = [];
      this.currentImage = null;
    },

    // 返回当前选中的项目（兼容旧代码调用）
    getCurrentProject(): Project | null {
      return this.currentProject;
    },

    setProjects(projects: Project[]) {
      this.projects = projects;
    },
    setCurrentProject(project: Project | null) {
      this.currentProject = project;
    },
    setCurrentImages(images: ImageAsset[]) {
      this.currentImages = images;
    },
    setCurrentImage(image: ImageAsset | null) {
      this.currentImage = image;
    }
  },
});
