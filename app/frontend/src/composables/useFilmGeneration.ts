import { FilmGenerationService } from '@/services/FilmGenerationService';
import { useProjectStore } from '@/stores/projectStore';
import { useEditorStore } from '@/stores/editorStore';

const filmService = new FilmGenerationService();

export function useFilmGeneration() {
  const projectStore = useProjectStore();
  const editorStore = useEditorStore();

  async function generate() {
    if (!projectStore.currentImage) return;

    editorStore.setLoading(true);
    try {
      const result = await filmService.generate(
        projectStore.currentImage.id,
        editorStore.basic,
        editorStore.film
      );
      editorStore.setResultUrl(result.result_url);
    } finally {
      editorStore.setLoading(false);
      alert('Film generation completed! You can now download the result image.');
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
