export interface BasicAdjustments {
  exposure: number;
  contrast: number;
  highlights: number;
  shadows: number;
  temperature: number;
  tint: number;
  saturation: number;
}

export interface DofSettings {
  enabled: boolean;
  focal_length_mm: number;
  f_number: number;
  focus_distance_m: number;
  sensor_width_mm: number;
  sensor_height_mm: number;
  depth_min_mm: number;
  depth_max_mm: number;
  psf_kernel_size: number;
  num_layers: number;
  render_method: string;
}

export interface FilmStyleSettings {
  preset: string;
  dof?: DofSettings;
}

export interface EditSession {
  imageId: string;
  basic: BasicAdjustments;
  film: FilmStyleSettings;
  resultUrl?: string;
}
