export interface BasicAdjustments {
  exposure: number;
  contrast: number;
  highlights: number;
  shadows: number;
  temperature: number;
  tint: number;
  saturation: number;
}

export interface FilmStyleSettings {
  preset: string;
  grain: number;
  halation: number;
}

export interface EditSession {
  imageId: string;
  basic: BasicAdjustments;
  film: FilmStyleSettings;
  resultUrl?: string;
}
