import { FilmGenerationService } from '@/services/FilmGenerationService';
import { useProjectStore } from '@/stores/projectStore';
import { useEditorStore } from '@/stores/editorStore';

const filmService = new FilmGenerationService();

export function useFilmGeneration() {
  const projectStore = useProjectStore();
  const editorStore = useEditorStore();

  async function generate() {
    if (!projectStore.currentImage) {
      editorStore.setGenerationError('请先选择一张图片。');
      return;
    }

    editorStore.setGenerationError('');
    editorStore.setGenerationStage('正在提交生成任务…');
    editorStore.setLoading(true);
    try {
      if (editorStore.film.dof?.enabled) {
        editorStore.setGenerationStage('正在计算景深…');
      } else {
        editorStore.setGenerationStage('正在应用胶片风格…');
      }

      const result = await filmService.generate(
        projectStore.currentImage.id,
        editorStore.basic,
        editorStore.film
      );
      editorStore.setGenerationStage('正在载入生成结果…');
      editorStore.setResultUrl(result.result_url);
    } catch (error: any) {
      console.error('Film generation failed:', error);
      editorStore.setGenerationError(
        error?.response?.data?.detail ?? error?.message ?? '胶片生成失败，请稍后重试。'
      );
    } finally {
      editorStore.setLoading(false);
    }
  }

  async function loadPresets() {
    return await filmService.getPresets();
  }

  return {
    generate,
    loadPresets,
  };
}
