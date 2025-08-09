'use client';

import React, { useCallback, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  FileText, 
  Image, 
  File, 
  X, 
  CheckCircle2, 
  AlertCircle,
  Loader2 
} from 'lucide-react';
import { UploadedFile } from '@/lib/types';

interface FileUploaderProps {
  onFilesSelect: (files: File[]) => void;
  acceptedTypes?: string[];
  maxFiles?: number;
  maxSize?: number; // in MB
}

export const FileUploader: React.FC<FileUploaderProps> = ({
  onFilesSelect,
  acceptedTypes = ['.pdf', '.docx', '.txt', '.jpg', '.jpeg', '.png'],
  maxFiles = 10,
  maxSize = 10
}) => {
  const [isDragActive, setIsDragActive] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  const getFileIcon = (fileName: string) => {
    const extension = fileName.toLowerCase().split('.').pop();
    switch (extension) {
      case 'pdf':
        return <FileText className="text-red-500" size={20} />;
      case 'docx':
      case 'doc':
        return <FileText className="text-blue-500" size={20} />;
      case 'txt':
        return <FileText className="text-gray-500" size={20} />;
      case 'jpg':
      case 'jpeg':
      case 'png':
        return <Image className="text-green-500" size={20} />;
      default:
        return <File className="text-gray-500" size={20} />;
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const validateFile = (file: File): string | null => {
    const extension = '.' + file.name.toLowerCase().split('.').pop();
    if (!acceptedTypes.includes(extension)) {
      return `File type ${extension} is not supported`;
    }
    if (file.size > maxSize * 1024 * 1024) {
      return `File size exceeds ${maxSize}MB limit`;
    }
    return null;
  };

  const processFiles = useCallback((files: File[]) => {
    const validFiles: File[] = [];

    files.forEach((file, index) => {
      const error = validateFile(file);
      
      if (!error && selectedFiles.length + validFiles.length < maxFiles) {
        validFiles.push(file);
      }
    });

    setSelectedFiles(prev => [...prev, ...validFiles]);
  }, [selectedFiles, maxFiles]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
    
    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
  }, [processFiles]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    processFiles(files);
    e.target.value = '';
  }, [processFiles]);

  const removeFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = () => {
    if (selectedFiles.length === 0) return;

    const newUploadedFiles: UploadedFile[] = selectedFiles.map(file => ({
      file,
      id: Math.random().toString(36).substr(2, 9),
      progress: 0,
      status: 'uploading' as const
    }));

    setUploadedFiles(prev => [...prev, ...newUploadedFiles]);
    
    // Simulate upload progress
    newUploadedFiles.forEach((uploadedFile, index) => {
      const interval = setInterval(() => {
        setUploadedFiles(prev => 
          prev.map(file => 
            file.id === uploadedFile.id 
              ? { ...file, progress: Math.min(file.progress + Math.random() * 30, 100) }
              : file
          )
        );
      }, 100);

      setTimeout(() => {
        clearInterval(interval);
        setUploadedFiles(prev => 
          prev.map(file => 
            file.id === uploadedFile.id 
              ? { ...file, progress: 100, status: 'success' }
              : file
          )
        );
      }, 1000 + index * 200);
    });

    onFilesSelect(selectedFiles);
    setSelectedFiles([]);
  };

  return (
    <div className="space-y-4">
      {/* Drop Zone */}
      <motion.div
        className={`
          relative border-2 border-dashed rounded-xl p-8 transition-all duration-200
          ${isDragActive 
            ? 'border-blue-500 bg-blue-50' 
            : 'border-slate-300 hover:border-slate-400 bg-slate-50 hover:bg-slate-100'
          }
        `}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <input
          type="file"
          multiple
          accept={acceptedTypes.join(',')}
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        
        <div className="text-center">
          <motion.div
            animate={{ y: isDragActive ? -5 : 0 }}
            transition={{ duration: 0.2 }}
            className={`
              w-16 h-16 rounded-full mx-auto mb-4 flex items-center justify-center
              ${isDragActive 
                ? 'bg-blue-500 text-white' 
                : 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
              }
            `}
          >
            <Upload size={24} />
          </motion.div>
          
          <h3 className="text-lg font-semibold text-slate-800 mb-2">
            {isDragActive ? 'Drop your files here' : 'Upload your documents'}
          </h3>
          
          <p className="text-slate-600 mb-4">
            Drag and drop files here, or click to browse
          </p>
          
          <div className="text-sm text-slate-500 space-y-1">
            <p>Supported formats: {acceptedTypes.join(', ')}</p>
            <p>Maximum file size: {maxSize}MB</p>
            <p>Maximum {maxFiles} files allowed</p>
          </div>
        </div>
      </motion.div>

      {/* Selected Files (Ready to Upload) */}
      <AnimatePresence>
        {selectedFiles.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-3"
          >
            <div className="flex items-center justify-between">
              <h4 className="font-medium text-slate-800">
                Selected Files ({selectedFiles.length})
              </h4>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleUpload}
                className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg font-medium hover:shadow-lg transition-all duration-200 flex items-center space-x-2"
              >
                <Upload size={16} />
                <span>Upload {selectedFiles.length} file{selectedFiles.length > 1 ? 's' : ''}</span>
              </motion.button>
            </div>
            
            <div className="space-y-2 max-h-48 overflow-y-auto scrollbar-thin">
              {selectedFiles.map((file, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="flex items-center space-x-3 p-3 bg-blue-50 border border-blue-200 rounded-lg"
                >
                  <div className="flex-shrink-0">
                    {getFileIcon(file.name)}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-slate-800 truncate">
                      {file.name}
                    </p>
                    <p className="text-xs text-slate-500">
                      {formatFileSize(file.size)}
                    </p>
                  </div>
                  
                  <button
                    onClick={() => removeFile(index)}
                    className="p-1 hover:bg-blue-100 rounded transition-colors"
                  >
                    <X size={14} className="text-slate-500" />
                  </button>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      {/* Uploaded Files List */}
      <AnimatePresence>
        {uploadedFiles.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-3"
          >
            <h4 className="font-medium text-slate-800">
              Uploading/Uploaded Files ({uploadedFiles.length})
            </h4>
            
            <div className="space-y-2 max-h-64 overflow-y-auto scrollbar-thin">
              {uploadedFiles.map((uploadedFile) => (
                <motion.div
                  key={uploadedFile.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="flex items-center space-x-3 p-3 bg-white rounded-lg border border-slate-200 shadow-sm"
                >
                  <div className="flex-shrink-0">
                    {getFileIcon(uploadedFile.file.name)}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-slate-800 truncate">
                      {uploadedFile.file.name}
                    </p>
                    <p className="text-xs text-slate-500">
                      {formatFileSize(uploadedFile.file.size)}
                    </p>
                    
                    {uploadedFile.status === 'uploading' && (
                      <div className="mt-1">
                        <div className="w-full bg-slate-200 rounded-full h-1.5">
                          <motion.div
                            className="bg-gradient-to-r from-blue-500 to-purple-600 h-1.5 rounded-full"
                            initial={{ width: 0 }}
                            animate={{ width: `${uploadedFile.progress}%` }}
                            transition={{ duration: 0.1 }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <div className="flex-shrink-0 flex items-center space-x-2">
                    {uploadedFile.status === 'uploading' && (
                      <Loader2 size={16} className="text-blue-500 animate-spin" />
                    )}
                    {uploadedFile.status === 'success' && (
                      <CheckCircle2 size={16} className="text-green-500" />
                    )}
                    {uploadedFile.status === 'error' && (
                      <AlertCircle size={16} className="text-red-500" />
                    )}
                    
                    <button
                      onClick={() => setUploadedFiles(prev => prev.filter(file => file.id !== uploadedFile.id))}
                      className="p-1 hover:bg-slate-100 rounded transition-colors"
                    >
                      <X size={14} className="text-slate-500" />
                    </button>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};