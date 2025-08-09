'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Plus, Loader2 } from 'lucide-react';
import { FileUploader } from './FileUploader';
import { KnowledgeBase } from '@/lib/types';
import toast from 'react-hot-toast';

interface AddDocumentsModalProps {
  isOpen: boolean;
  knowledgeBase: KnowledgeBase | null;
  onClose: () => void;
  onSubmit: (files: File[]) => Promise<void>;
}

export const AddDocumentsModal: React.FC<AddDocumentsModalProps> = ({
  isOpen,
  knowledgeBase,
  onClose,
  onSubmit,
}) => {
  const [files, setFiles] = useState<File[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (files.length === 0) {
      toast.error('Please select at least one document');
      return;
    }

    setIsSubmitting(true);
    try {
      await onSubmit(files);
      toast.success(`Added ${files.length} document${files.length > 1 ? 's' : ''} successfully!`);
      handleClose();
    } catch (error) {
      toast.error('Failed to add documents');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    if (!isSubmitting) {
      setFiles([]);
      onClose();
    }
  };

  return (
    <AnimatePresence>
      {isOpen && knowledgeBase && (
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
                <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-emerald-500 to-teal-600 flex items-center justify-center">
                  <Plus size={20} className="text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-slate-900">Add Documents</h2>
                  <p className="text-sm text-slate-600">
                    Add more documents to "{knowledgeBase.name}"
                  </p>
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
              {/* Knowledge Base Info */}
              <div className="flex items-center space-x-3 p-4 bg-slate-50 rounded-xl">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center text-lg">
                  {knowledgeBase.icon}
                </div>
                <div>
                  <h3 className="font-semibold text-slate-900">{knowledgeBase.name}</h3>
                  <p className="text-sm text-slate-600">
                    Currently has {knowledgeBase.documentCount} documents
                  </p>
                </div>
              </div>

              {/* File Uploader */}
              <div>
                <label className="block text-sm font-medium text-slate-800 mb-2">
                  Select Documents to Add
                </label>
                <FileUploader
                  onFilesSelect={setFiles}
                  maxFiles={15}
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
                  disabled={isSubmitting || files.length === 0}
                  className="px-6 py-2.5 bg-gradient-to-r from-emerald-500 to-teal-600 text-white rounded-xl font-medium hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 size={18} className="animate-spin" />
                      <span>Adding...</span>
                    </>
                  ) : (
                    <span>Add Documents ({files.length})</span>
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