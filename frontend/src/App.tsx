
import { useCallback, useMemo, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Document, Page, pdfjs } from "react-pdf";
import {
  FileText,
  FileUp,
  MessageSquare,
  Send,
  Trash2,
  PanelLeftClose,
  PanelRightClose,
  ShieldCheck,
  Bot,
  UserRound,
  LoaderCircle,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  ExternalLink
} from "lucide-react";

import { askQuestion, isDemoMode, uploadDocument } from "./api";
import type { ChatMessage, DocumentItem, Source } from "./types";

import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

const formatSize = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const initialMessage: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content:
    "Hallo! Ich bin dein Legal RAG Assistant. Lade links Dokumente hoch und stelle anschließend eine Frage dazu. Im aktuellen Demo-Modus sind Antworten noch Beispiele und keine echte Dokumentenanalyse."
};

export default function App() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([initialMessage]);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(
    null
  );
  const [pageNumber, setPageNumber] = useState(1);
  const [numPages, setNumPages] = useState(0);
  const [question, setQuestion] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);

  const selectedDocument = useMemo(
    () => documents.find((doc) => doc.id === selectedDocumentId) ?? null,
    [documents, selectedDocumentId]
  );

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (!acceptedFiles.length) return;

    setError(null);
    setIsUploading(true);

    try {
      const uploaded = await Promise.all(
        acceptedFiles.map((file) => uploadDocument(file))
      );

      setDocuments((current) => [...uploaded, ...current]);
      setSelectedDocumentId((current) => current ?? uploaded[0].id);
      setPageNumber(1);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Upload fehlgeschlagen."
      );
    } finally {
      setIsUploading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    noClick: true,
    multiple: true,
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        [".docx"],
      "text/plain": [".txt"]
    }
  });

  function removeDocument(id: string) {
    const documentToRemove = documents.find((doc) => doc.id === id);

    if (documentToRemove?.fileUrl.startsWith("blob:")) {
      URL.revokeObjectURL(documentToRemove.fileUrl);
    }

    setDocuments((current) => current.filter((doc) => doc.id !== id));

    if (selectedDocumentId === id) {
      const replacement = documents.find((doc) => doc.id !== id);
      setSelectedDocumentId(replacement?.id ?? null);
      setPageNumber(1);
    }
  }

  function openSource(source: Source) {
    setSelectedDocumentId(source.documentId);
    setPageNumber(source.page ?? 1);
    setRightOpen(true);
  }

  async function sendMessage() {
    const text = question.trim();
    if (!text || isAsking) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text
    };

    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setQuestion("");
    setIsAsking(true);
    setError(null);

    try {
      const result = await askQuestion(text, documents, nextMessages);

      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: result.answer,
          sources: result.sources
        }
      ]);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Die Anfrage ist fehlgeschlagen."
      );
    } finally {
      setIsAsking(false);
    }
  }

  return (
    <main className="app-shell" {...getRootProps()}>
      <input {...getInputProps()} />

      {isDragActive && (
        <div className="drop-overlay">
          <FileUp size={36} />
          <strong>Dokumente hier ablegen</strong>
          <span>PDF, DOCX oder TXT</span>
        </div>
      )}

      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><ShieldCheck size={21} /></div>
          <div>
            <div className="brand-title">Legal RAG Assistant</div>
            <div className="brand-subtitle">Dokumente verstehen. Quellen nachvollziehen.</div>
          </div>
        </div>

        <div className="topbar-right">
          {isDemoMode() && <span className="demo-badge">Demo-Modus</span>}
          <span className="privacy-note">
            <span className="status-dot" />
            Lokale Vorschau
          </span>
        </div>
      </header>

      <div className="workspace">
        {leftOpen && (
          <aside className="sidebar left-sidebar">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">WORKSPACE</span>
                <h2>Dokumente</h2>
              </div>
              <button className="icon-button" onClick={() => setLeftOpen(false)} title="Seitenleiste schließen">
                <PanelLeftClose size={17} />
              </button>
            </div>

            <button className="upload-zone" onClick={open}>
              <span className="upload-icon"><FileUp size={21} /></span>
              <strong>Dokumente hinzufügen</strong>
              <span>Drag & Drop oder Datei auswählen</span>
              <small>PDF · DOCX · TXT</small>
            </button>

            {isUploading && (
              <div className="upload-status">
                <LoaderCircle className="spin" size={15} />
                Dokumente werden verarbeitet …
              </div>
            )}

            <div className="document-list-heading">
              <span>DATEIEN</span>
              <span className="count">{documents.length}</span>
            </div>

            <div className="document-list">
              {documents.length === 0 ? (
                <div className="empty-documents">
                  <FileText size={25} />
                  <p>Noch keine Dokumente</p>
                  <span>Füge Dokumente hinzu, um sie hier zu sehen.</span>
                </div>
              ) : (
                documents.map((doc) => (
                  <div
                    key={doc.id}
                    className={`document-item ${
                      doc.id === selectedDocumentId ? "selected" : ""
                    }`}
                  >
                    <button
                      className="document-select"
                      onClick={() => {
                        setSelectedDocumentId(doc.id);
                        setPageNumber(1);
                      }}
                    >
                      <span className="file-icon"><FileText size={18} /></span>
                      <span className="document-meta">
                        <strong title={doc.name}>{doc.name}</strong>
                        <small>{formatSize(doc.size)}</small>
                      </span>
                    </button>
                    <button
                      className="remove-button"
                      title="Dokument entfernen"
                      onClick={() => removeDocument(doc.id)}
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                ))
              )}
            </div>

            <div className="sidebar-footer">
              <span className="footer-icon"><ShieldCheck size={16} /></span>
              <span>Demo mit synthetischen Dokumenten testen. Keine vertraulichen Dateien verwenden.</span>
            </div>
          </aside>
        )}

        <section className="chat-panel">
          <div className="chat-heading">
            {!leftOpen && (
              <button className="icon-button" onClick={() => setLeftOpen(true)} title="Dokumente öffnen">
                <PanelLeftClose size={17} />
              </button>
            )}
            <div className="chat-heading-icon"><MessageSquare size={17} /></div>
            <div>
              <h2>Chat</h2>
              <span>{documents.length} Dokumente im Workspace</span>
            </div>
            {!rightOpen && (
              <button className="icon-button push-right" onClick={() => setRightOpen(true)} title="Dokumentvorschau öffnen">
                <PanelRightClose size={17} />
              </button>
            )}
          </div>

          <div className="messages">
            {messages.map((message) => (
              <article key={message.id} className={`message ${message.role}`}>
                <div className="avatar">
                  {message.role === "assistant" ? <Bot size={18} /> : <UserRound size={18} />}
                </div>
                <div className="message-body">
                  <div className="message-author">
                    {message.role === "assistant" ? "Assistant" : "Du"}
                  </div>
                  <div className="message-text">{message.content}</div>

                  {message.sources && message.sources.length > 0 && (
                    <div className="sources">
                      <div className="sources-heading">QUELLEN</div>
                      {message.sources.map((source, index) => (
                        <button
                          key={`${source.documentId}-${index}`}
                          className="source-card"
                          onClick={() => openSource(source)}
                        >
                          <span className="source-number">{index + 1}</span>
                          <span className="source-info">
                            <strong>{source.documentName}</strong>
                            <small>
                              {source.page ? `Seite ${source.page}` : "Dokument"}
                              {source.excerpt ? ` · ${source.excerpt}` : ""}
                            </small>
                          </span>
                          <ExternalLink size={15} />
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </article>
            ))}

            {isAsking && (
              <div className="thinking">
                <LoaderCircle className="spin" size={16} />
                Antwort wird erstellt …
              </div>
            )}
          </div>

          {error && (
            <div className="error-message">
              <AlertCircle size={16} />
              <span>{error}</span>
              <button onClick={() => setError(null)}>Schließen</button>
            </div>
          )}

          <div className="composer-wrap">
            <form
              className="composer"
              onSubmit={(event) => {
                event.preventDefault();
                void sendMessage();
              }}
            >
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Stelle eine Frage zu deinen Dokumenten …"
                rows={2}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void sendMessage();
                  }
                }}
              />
              <div className="composer-bottom">
                <span>Enter zum Senden · Shift + Enter für neue Zeile</span>
                <button type="submit" disabled={!question.trim() || isAsking} className="send-button">
                  <Send size={15} />
                  Senden
                </button>
              </div>
            </form>
            <p className="disclaimer">Antworten können Fehler enthalten. Prüfe wichtige Aussagen anhand der Originalquellen.</p>
          </div>
        </section>

        {rightOpen && (
          <aside className="preview-panel">
            <div className="preview-heading">
              <div>
                <span className="eyebrow">ORIGINALQUELLE</span>
                <h2>Dokumentvorschau</h2>
              </div>
              <button className="icon-button" onClick={() => setRightOpen(false)} title="Vorschau schließen">
                <PanelRightClose size={17} />
              </button>
            </div>

            {selectedDocument ? (
              <>
                <div className="preview-file-title">
                  <FileText size={17} />
                  <span title={selectedDocument.name}>{selectedDocument.name}</span>
                </div>

                {selectedDocument.name.toLowerCase().endsWith(".pdf") ? (
                  <div className="pdf-stage">
                    <Document
                      file={selectedDocument.fileUrl}
                      onLoadSuccess={({ numPages: total }) => setNumPages(total)}
                      onLoadError={() => setError("PDF konnte nicht geöffnet werden.")}
                      loading={<div className="preview-empty">PDF wird geladen …</div>}
                      error={<div className="preview-empty">PDF konnte nicht geladen werden.</div>}
                    >
                      <Page
                        pageNumber={pageNumber}
                        width={Math.min(560, window.innerWidth * 0.38)}
                        renderTextLayer
                        renderAnnotationLayer
                      />
                    </Document>
                  </div>
                ) : (
                  <div className="preview-empty">
                    <FileText size={32} />
                    <strong>Keine PDF-Vorschau verfügbar</strong>
                    <span>Die Vorschau unterstützt derzeit PDF-Dateien. DOCX und TXT werden in der Dokumentenliste angezeigt.</span>
                    {selectedDocument.fileUrl && (
                      <a href={selectedDocument.fileUrl} target="_blank" rel="noreferrer">
                        Datei separat öffnen
                      </a>
                    )}
                  </div>
                )}

                {selectedDocument.name.toLowerCase().endsWith(".pdf") && (
                  <div className="pdf-controls">
                    <button
                      className="icon-button"
                      disabled={pageNumber <= 1}
                      onClick={() => setPageNumber((page) => Math.max(1, page - 1))}
                      title="Vorherige Seite"
                    >
                      <ChevronLeft size={18} />
                    </button>
                    <span>
                      Seite <strong>{pageNumber}</strong>
                      {numPages > 0 ? ` von ${numPages}` : ""}
                    </span>
                    <button
                      className="icon-button"
                      disabled={numPages > 0 && pageNumber >= numPages}
                      onClick={() => setPageNumber((page) => Math.min(numPages || page + 1, page + 1))}
                      title="Nächste Seite"
                    >
                      <ChevronRight size={18} />
                    </button>
                  </div>
                )}
              </>
            ) : (
              <div className="preview-empty no-document">
                <FileText size={34} />
                <strong>Wähle ein Dokument aus</strong>
                <span>Originaldokumente und später auch die gefundenen Quellseiten erscheinen hier.</span>
              </div>
            )}
          </aside>
        )}
      </div>
    </main>
  );
}
