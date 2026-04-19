export type TaskStatus = 'idle' | 'pending' | 'running' | 'done' | 'failed';

export interface GenerationTask {
  id: string;
  imageId: string;
  status: TaskStatus;
  resultUrl?: string;
  errorMessage?: string;
  createdAt: string;
}
