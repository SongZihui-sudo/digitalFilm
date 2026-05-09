import type { BasicAdjustments } from '@/models/edit'
import { apiClient } from '@/api/client'

export interface ImageEditSettings extends BasicAdjustments {
  imageId: string
  preset?: string
  grain?: number
  halation?: number
}

export class ImageEditorService {
  createDefaultAdjustments(): BasicAdjustments {
    return {
      exposure: 0,
      contrast: 0,
      highlights: 0,
      shadows: 0,
      temperature: 0,
      tint: 0,
      saturation: 0,
    }
  }

  createDefaultSettings(imageId: string): ImageEditSettings {
    return {
      imageId,
      ...this.createDefaultAdjustments(),
      preset: '',
      grain: 0,
      halation: 0,
    }
  }

  async getImageEditSettings(imageId: string): Promise<ImageEditSettings> {
    try {
      const { data } = await apiClient.get(`/api/images/${imageId}/settings`)
      return {
        ...this.createDefaultSettings(imageId),
        ...data,
      }
    } catch (error) {
      console.error('get image edit settings failed:', error)
      return this.createDefaultSettings(imageId)
    }
  }

  async updateImageEditSettings(
    imageId: string,
    settings: Partial<ImageEditSettings>,
  ): Promise<ImageEditSettings> {
    const payload = {
      ...this.createDefaultSettings(imageId),
      ...settings,
      imageId,
    }

    const { data } = await apiClient.post(`/api/images/${imageId}/settings`, payload)

    return data.data ?? data
  }

  async deleteImage(imageID: string) {
    // 请求变成: DELETE /api/images/456
    const { data } = await apiClient.delete(`/api/images/${imageID}`);
    return data; // 建议也 return data，保持风格一致
  }
}

export const imageEditorService = new ImageEditorService()
