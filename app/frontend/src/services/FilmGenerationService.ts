import { apiClient, image_client } from '@/api/client';
import type { BasicAdjustments, FilmStyleSettings } from '@/models/edit';

export class FilmGenerationService {
  async generate(
    imageId: string,
    basic: BasicAdjustments,
    film: FilmStyleSettings
  ) {
    const { data } = await image_client.post('/api/film/generate', {
      image_id: imageId,
      basic,
      film,
    });
    return data;
  }

  async getPresets() {
    const { data } = await apiClient.get('/api/film/presets');
    return data;
  }
}
