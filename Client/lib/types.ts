export interface KnowledgeBase {
  id: string;
  name: string;
  icon: string;
  lastUpdated: string;
  documentCount: number;
  createdAt: string;
}

export interface ChatMessage {
  id: string;
  content: string;
  type: 'user' | 'ai';
  timestamp: string;
  isLoading?: boolean;
}

export interface Document {
  id: string;
  name: string;
  type: string;
  size: number;
  uploadedAt: string;
}

export interface UploadedFile {
  file: File;
  id: string;
  progress: number;
  status: 'uploading' | 'success' | 'error';
}