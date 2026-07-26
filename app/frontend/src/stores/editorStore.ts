import { defineStore } from 'pinia';
import type { BasicAdjustments, FilmStyleSettings } from '@/models/edit';

interface EditorState {
  basic: BasicAdjustments;
  film: FilmStyleSettings;
  previewUrl: string;
  resultUrl: string;
  resultBasic: BasicAdjustments | null;
  loading: boolean;
  generationStage: string;
  generationError: string;
}

export const useEditorStore = defineStore('editor', {
  state: (): EditorState => ({
    basic: {
      exposure: 0,
      contrast: 0,
      highlights: 0,
      shadows: 0,
      temperature: 0,
      tint: 0,
      saturation: 0,
    },
    film: {
      preset: 'kodak_gold_200',
      dof: {
        enabled: false,
        focal_length_mm: 210,
        f_number: 5.6,
        focus_distance_m: 2.5,
        sensor_width_mm: 127,
        sensor_height_mm: 101.6,
        depth_min_mm: 500,
        depth_max_mm: 12000,
        psf_kernel_size: 65,
        num_layers: 8,
        render_method: 'psf_patch',
      },
    },
    previewUrl: '',
    resultUrl: '',
    resultBasic: null,
    loading: false,
    generationStage: '',
    generationError: '',
  }),

  actions: {
    setBasic<K extends keyof BasicAdjustments>(key: K, value: BasicAdjustments[K]) {
      this.basic[key] = value;
    },
    setFilm<K extends keyof FilmStyleSettings>(key: K, value: FilmStyleSettings[K]) {
      this.film[key] = value;
    },
    setPreviewUrl(url: string) {
      this.previewUrl = url;
    },
    setResultUrl(url: string) {
      this.resultUrl = url;
      this.resultBasic = { ...this.basic };
    },
    setLoading(v: boolean) {
      this.loading = v;
      if (!v) this.generationStage = '';
    },
    setGenerationStage(stage: string) {
      this.generationStage = stage;
    },
    setGenerationError(error: string) {
      this.generationError = error;
    },
    resetEditor() {
      this.basic = {
        exposure: 0,
        contrast: 0,
        highlights: 0,
        shadows: 0,
        temperature: 0,
        tint: 0,
        saturation: 0,
      };
      this.film = {
        preset: 'kodak_gold_200',
        dof: {
          enabled: false,
          focal_length_mm: 210,
          f_number: 5.6,
          focus_distance_m: 2.5,
          sensor_width_mm: 127,
          sensor_height_mm: 101.6,
          depth_min_mm: 500,
          depth_max_mm: 12000,
          psf_kernel_size: 129,
          num_layers: 24,
        },
      };
      this.resultUrl = '';
      this.resultBasic = null;
    },
  },
});
