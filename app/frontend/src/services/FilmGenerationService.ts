import { apiClient, image_client } from '@/api/client';
import type { BasicAdjustments, FilmStyleSettings } from '@/models/edit';

// CPU 光学模拟（尤其是 Depth Anything 与大画幅散焦）可能远超普通 API 的 30 秒。
// 仅放宽生成请求，其他图像服务请求仍保留默认超时，避免掩盖常规接口故障。
const FILM_GENERATION_TIMEOUT_MS = 10 * 60 * 1000;

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
    }, {
      timeout: FILM_GENERATION_TIMEOUT_MS,
    });
    return data;
  }

  async getPresets() {
    const { data } = await apiClient.get('/api/film/presets');
    return data;
  }
}
