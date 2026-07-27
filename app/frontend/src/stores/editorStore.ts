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
        f_number: 5.6,
        focal_length_mm: 85,
        sensor_profile: 'full_frame',
        focus_point_x: null,
        focus_point_y: null,
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
          f_number: 5.6,
          focal_length_mm: 85,
          sensor_profile: 'full_frame',
          focus_point_x: null,
          focus_point_y: null,
        },
      };
      this.resultUrl = '';
      this.resultBasic = null;
    },
  },
});
