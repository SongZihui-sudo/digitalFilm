export interface ImageAsset {
  id: string;
  projectId: string;
  name: string;
  originalUrl: string;
  thumbnailUrl?: string;
  width?: number;
  height?: number;
  createdAt: string;
}
