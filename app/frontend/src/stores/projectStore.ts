import { defineStore } from 'pinia';
import type { Project } from '@/models/project';
import type { ImageAsset } from '@/models/image';

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
    },
    getCurrentProject() {
      return this.currentProject;
    }
  },
});
