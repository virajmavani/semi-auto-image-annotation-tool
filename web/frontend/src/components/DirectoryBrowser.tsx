import { useState, useEffect } from 'react';
import { ChevronRight, Folder, FolderOpen, ArrowUp, X, Check } from 'lucide-react';
import { api } from '../api/client';

interface DirectoryEntry {
  name: string;
  path: string;
}

interface BrowseResult {
  current_path: string;
  parent_path: string | null;
  entries: DirectoryEntry[];
}

interface Props {
  initialPath?: string;
  onSelect: (path: string) => void;
  onClose: () => void;
  theme: 'dark' | 'light';
}

const themeClasses = {
  dark: {
    overlay: 'bg-black/60',
    modal: 'bg-slate-800 border-slate-600',
    header: 'bg-slate-900 border-slate-700',
    pathBar: 'bg-slate-900 border-slate-700 text-slate-300',
    row: 'hover:bg-slate-700 text-slate-100',
    rowSelected: 'bg-blue-700 text-white',
    border: 'border-slate-700',
    text: 'text-slate-100',
    textSecondary: 'text-slate-400',
    empty: 'text-slate-500',
    footer: 'bg-slate-900 border-slate-700',
  },
  light: {
    overlay: 'bg-black/40',
    modal: 'bg-white border-gray-300',
    header: 'bg-gray-50 border-gray-300',
    pathBar: 'bg-gray-100 border-gray-300 text-gray-700',
    row: 'hover:bg-blue-50 text-gray-900',
    rowSelected: 'bg-blue-100 text-blue-800',
    border: 'border-gray-200',
    text: 'text-gray-900',
    textSecondary: 'text-gray-500',
    empty: 'text-gray-400',
    footer: 'bg-gray-50 border-gray-300',
  },
};

export function DirectoryBrowser({ initialPath, onSelect, onClose, theme }: Props) {
  const t = themeClasses[theme];
  const [browseResult, setBrowseResult] = useState<BrowseResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedEntry, setSelectedEntry] = useState<DirectoryEntry | null>(null);

  const navigate = async (path: string) => {
    setLoading(true);
    setError(null);
    setSelectedEntry(null);
    try {
      const result = await api.browseDirectory(path);
      setBrowseResult(result);
    } catch (e: any) {
      setError(e.response?.data?.detail || 'Failed to read directory');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    navigate(initialPath || '~');
  }, []);

  const handleRowClick = (entry: DirectoryEntry) => {
    setSelectedEntry(prev => prev?.path === entry.path ? null : entry);
  };

  const handleRowDoubleClick = (entry: DirectoryEntry) => {
    navigate(entry.path);
  };

  const handleSelect = () => {
    if (selectedEntry) {
      onSelect(selectedEntry.path);
    } else if (browseResult) {
      onSelect(browseResult.current_path);
    }
  };

  const currentPath = browseResult?.current_path ?? '';
  const displayPath = selectedEntry ? selectedEntry.path : currentPath;

  return (
    <div className={`fixed inset-0 ${t.overlay} flex items-center justify-center z-50`}>
      <div className={`${t.modal} border rounded-xl shadow-2xl flex flex-col`} style={{ width: 560, height: 480 }}>
        {/* Header */}
        <div className={`flex items-center justify-between px-4 py-3 ${t.header} border-b ${t.border} rounded-t-xl`}>
          <span className={`font-semibold ${t.text}`}>Select Directory</span>
          <button onClick={onClose} className={`${t.textSecondary} hover:${t.text} transition-colors`}>
            <X size={18} />
          </button>
        </div>

        {/* Navigation bar */}
        <div className={`flex items-center gap-2 px-3 py-2 ${t.header} border-b ${t.border}`}>
          <button
            onClick={() => browseResult?.parent_path && navigate(browseResult.parent_path)}
            disabled={!browseResult?.parent_path || loading}
            className={`p-1 rounded ${t.textSecondary} disabled:opacity-40 hover:opacity-80 transition-opacity`}
            title="Go up"
          >
            <ArrowUp size={16} />
          </button>
          <div className={`flex-1 px-2 py-1 text-sm rounded border ${t.pathBar} truncate font-mono`}>
            {displayPath || '…'}
          </div>
        </div>

        {/* Directory listing */}
        <div className="flex-1 overflow-y-auto">
          {loading && (
            <div className={`flex items-center justify-center h-full ${t.textSecondary} text-sm`}>
              Loading…
            </div>
          )}
          {error && (
            <div className="flex items-center justify-center h-full text-red-500 text-sm px-4 text-center">
              {error}
            </div>
          )}
          {!loading && !error && browseResult && (
            browseResult.entries.length === 0 ? (
              <div className={`flex items-center justify-center h-full ${t.empty} text-sm`}>
                No subdirectories
              </div>
            ) : (
              <ul className="py-1">
                {browseResult.entries.map(entry => {
                  const isSelected = selectedEntry?.path === entry.path;
                  return (
                    <li
                      key={entry.path}
                      onClick={() => handleRowClick(entry)}
                      onDoubleClick={() => handleRowDoubleClick(entry)}
                      className={`flex items-center gap-3 px-4 py-2 cursor-pointer select-none transition-colors ${
                        isSelected ? t.rowSelected : t.row
                      }`}
                    >
                      {isSelected
                        ? <FolderOpen size={16} className="shrink-0 text-blue-300" />
                        : <Folder size={16} className="shrink-0 text-blue-400" />
                      }
                      <span className="text-sm truncate">{entry.name}</span>
                      <ChevronRight size={14} className={`ml-auto shrink-0 ${t.textSecondary}`} />
                    </li>
                  );
                })}
              </ul>
            )
          )}
        </div>

        {/* Footer */}
        <div className={`flex items-center justify-between gap-3 px-4 py-3 ${t.footer} border-t ${t.border} rounded-b-xl`}>
          <span className={`text-xs ${t.textSecondary} truncate flex-1`}>
            {selectedEntry
              ? `Selected: ${selectedEntry.name}`
              : 'Double-click to open, or click to select'}
          </span>
          <div className="flex gap-2 shrink-0">
            <button
              onClick={onClose}
              className={`px-3 py-1.5 text-sm border ${t.border} rounded-lg ${t.textSecondary} hover:opacity-80 transition-opacity`}
            >
              Cancel
            </button>
            <button
              onClick={handleSelect}
              disabled={!browseResult}
              className="px-3 py-1.5 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center gap-1.5 disabled:opacity-50 transition-colors"
            >
              <Check size={14} />
              Select
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
