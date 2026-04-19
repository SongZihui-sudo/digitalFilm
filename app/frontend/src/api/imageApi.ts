import { apiClient } from './client'
import { static_client } from './client'

export const imageApi = {
  async uploadImage(file: File, projectId: string) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('project_id', projectId)

    const { data } = await static_client.post('/api/images/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })

    return data
  },

  async listProjectImages(projectId: string) {
    const { data } = await apiClient.get(`/api/projects/${projectId}/images`)
    return data
  },
}
