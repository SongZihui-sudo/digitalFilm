import { ProjectService } from '@/services/ProjectService'
import { useProjectStore } from '@/stores/projectStore'
import { imageApi } from '@/api/imageApi'

const projectService = new ProjectService()

export function useProjectManager() {
  const projectStore = useProjectStore()

  async function loadProjects() {
    try {
      const projects = await projectService.listProjects()
      projectStore.setProjects(projects)

      if (projects.length > 0) {
        projectStore.setCurrentProject(projects[0])
        await loadProjectImages(projects[0].id)
      }
    } catch (error) {
      console.error('loadProjects failed:', error)
    }
  }

  async function createProject(name: string) {
    try {
      const project = await projectService.createProject(name)
      await loadProjects()
      projectStore.setCurrentProject(project)
    } catch (error) {
      console.error('createProject failed:', error)
    }
  }

  async function loadProjectImages(projectId: string) {
    try {
      const images = await imageApi.listProjectImages(projectId)
      projectStore.setCurrentImages(images)
      if (images.length > 0) {
        projectStore.setCurrentImage(images[0])
      }
    } catch (error) {
      console.error('loadProjectImages failed:', error)
    }
  }

  async function uploadImage(file: File) {
    if (!projectStore.currentProject) {
      alert('请先选择或创建一个项目')
      return
    }

    try {
      await imageApi.uploadImage(file, projectStore.currentProject.id)
      alert('上传成功')
      await loadProjectImages(projectStore.currentProject.id)
    } catch (error) {
      console.error('uploadImage failed:', error)
      alert('上传失败')
    }
  }

  return {
    loadProjects,
    createProject,
    loadProjectImages,
    uploadImage,
  }
}
