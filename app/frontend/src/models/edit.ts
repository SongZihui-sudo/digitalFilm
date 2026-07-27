export interface BasicAdjustments {
  exposure: number;
  contrast: number;
  highlights: number;
  shadows: number;
  temperature: number;
  tint: number;
  saturation: number;
}

/** 传感器尺寸预设 */
export type SensorSizeProfile = 'm43' | 'aps_c' | 'full_frame' | 'medium_645' | 'large_8x10';

/** 传感器尺寸预设 → 物理尺寸 (mm) 映射 */
export const SENSOR_SIZE_MAP: Record<SensorSizeProfile, { width: number; height: number }> = {
  m43:           { width: 17.3, height: 13.0 },
  aps_c:         { width: 23.6, height: 15.7 },
  full_frame:    { width: 36.0, height: 24.0 },
  medium_645:    { width: 56.0, height: 41.5 },
  large_8x10:    { width: 203.2, height: 254.0 },
};

export const SENSOR_SIZE_LABELS: Record<SensorSizeProfile, string> = {
  m43:        '4/3 英寸 (M4/3)',
  aps_c:      'APS-C (半画幅)',
  full_frame: '全画幅 (135)',
  medium_645: '中画幅 (645)',
  large_8x10: '大画幅 (8×10)',
};

/** 精简后的景深设置
 *  - 光圈值 (f_number) 控制虚化强度
 *  - 焦距 (focal_length_mm) 控制焦段
 *  - 传感器尺寸 (sensor_profile) 预设
 *  - 在预览图上点击设置对焦点 (focus_point_x/y, 归一化 0-1)
 *  - 其余参数固定为合理默认值
 */
export interface DofSettings {
  enabled: boolean;
  f_number: number;
  focal_length_mm: number;
  sensor_profile: SensorSizeProfile;
  /** 对焦点 X (归一化 0-1, null 表示默认中心) */
  focus_point_x: number | null;
  /** 对焦点 Y (归一化 0-1, null 表示默认中心) */
  focus_point_y: number | null;
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
