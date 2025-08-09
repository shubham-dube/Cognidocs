'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Database, Loader2 } from 'lucide-react';
import { FileUploader } from './FileUploader';
import toast from 'react-hot-toast';

interface CreateKnowledgeBaseModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (name: string, files: File[]) => Promise<void>;
}

export const CreateKnowledgeBaseModal: React.FC<CreateKnowledgeBaseModalProps> = ({
  isOpen,
  onClose,
  onSubmit,
}) => {
  const [name, setName] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!name.trim()) {
      toast.error('Please enter a knowledge base name');
      return;
    }
    
    if (files.length === 0) {
      toast.error('Please upload at least one document');
      return;
    }

    setIsSubmitting(true);
    try {
      await onSubmit(name.trim(), files);
      toast.success('Knowledge base created successfully!');
      handleClose();
    } catch (error) {
      toast.error('Failed to create knowledge base');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    if (!isSubmitting) {
      setName('');
      setFiles([]);
      onClose();
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50"
          onClick={handleClose}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-slate-200">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center">
                  <Database size={20} className="text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-slate-900">Create Knowledge Base</h2>
                  <p className="text-sm text-slate-600">Upload documents to create your AI assistant</p>
                </div>
              </div>
              
              <button
                onClick={handleClose}
                disabled={isSubmitting}
                className="p-2 hover:bg-slate-100 rounded-lg transition-colors disabled:opacity-50"
              >
                <X size={20} className="text-slate-500" />
              </button>
            </div>

            {/* Content */}
            <form onSubmit={handleSubmit} className="p-6 space-y-6">
              {/* Name Input */}
              <div>
                <label className="block text-sm font-medium text-slate-800 mb-2">
                  Knowledge Base Name *
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g., Health Insurance Policy, Company Handbook..."
                  disabled={isSubmitting}
                  className="w-full px-4 py-3 border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all placeholder-slate-400 disabled:opacity-50 disabled:cursor-not-allowed"
                />
              </div>

              {/* File Uploader */}
              <div>
                <label className="block text-sm font-medium text-slate-800 mb-2">
                  Upload Documents *
                </label>
                <FileUploader
                  onFilesSelect={setFiles}
                  maxFiles={20}
                  maxSize={25}
                />
              </div>

              {/* Actions */}
              <div className="flex items-center justify-end space-x-3 pt-4 border-t border-slate-200">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  type="button"
                  onClick={handleClose}
                  disabled={isSubmitting}
                  className="px-6 py-2.5 text-slate-600 hover:text-slate-800 transition-colors disabled:opacity-50"
                >
                  Cancel
                </motion.button>
                
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  type="submit"
                  disabled={isSubmitting || !name.trim() || files.length === 0}
                  className="px-6 py-2.5 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-medium hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 size={18} className="animate-spin" />
                      <span>Creating...</span>
                    </>
                  ) : (
                    <span>Create Knowledge Base</span>
                  )}
                </motion.button>
              </div>
            </form>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};