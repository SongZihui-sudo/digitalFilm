import { defineStore } from 'pinia';
import type { BasicAdjustments, FilmStyleSettings } from '@/models/edit';

interface EditorState {
  basic: BasicAdjustments;
  film: FilmStyleSettings;
  previewUrl: string;
  resultUrl: string;
  loading: boolean;
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
      grain: 0,
      halation: 0,
    },
    previewUrl: '',
    resultUrl: '',
    loading: false,
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
    },
    setLoading(v: boolean) {
      this.loading = v;
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
        grain: 0,
        halation: 0,
      };
      this.resultUrl = '';
    },
  },
});
