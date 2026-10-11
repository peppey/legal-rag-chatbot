
export type Source = {
  documentId: string;
  documentName: string;
  page?: number;
  excerpt?: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

export type DocumentItem = {
  id: string;
  name: string;
  type: string;
  size: number;
  uploadedAt: string;
  fileUrl: string;
  file?: File;
  pageCount?: number;
};
