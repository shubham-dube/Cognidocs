'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Plus, 
  Search, 
  ChevronLeft, 
  ChevronRight,
  Database,
  Sparkles
} from 'lucide-react';
import { KnowledgeBase } from '@/lib/types';
import { KnowledgeBaseItem } from './KnowledgeBaseItem';
import { KnowledgeBaseSkeletons } from '../ui/LoadingSkeleton';

interface SidebarProps {
  knowledgeBases: KnowledgeBase[];
  isLoading: boolean;
  selectedKbId: string | null;
  isCollapsed: boolean;
  searchTerm: string;
  onSelectKb: (kbId: string) => void;
  onToggleCollapse: () => void;
  onSearchChange: (term: string) => void;
  onCreateKb: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  knowledgeBases,
  isLoading,
  selectedKbId,
  isCollapsed,
  searchTerm,
  onSelectKb,
  onToggleCollapse,
  onSearchChange,
  onCreateKb,
}) => {
  const filteredKbs = knowledgeBases.filter(kb =>
    kb.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <motion.div
      initial={{ width: 320 }}
      animate={{ width: isCollapsed ? 80 : 320 }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="h-full bg-white border-r border-slate-200 shadow-lg flex flex-col relative"
    >
      {/* Header */}
      <div className="p-4 border-b border-slate-200">
        <AnimatePresence mode="wait">
          {!isCollapsed ? (
            <motion.div
              key="expanded-header"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.2 }}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center">
                    <Sparkles size={18} className="text-white" />
                  </div>
                  <h1 className="text-xl font-bold text-slate-900">DocuChat</h1>
                </div>
                <button
                  onClick={onToggleCollapse}
                  className="p-1.5 rounded-lg hover:bg-slate-100 transition-colors"
                >
                  <ChevronLeft size={18} className="text-slate-600" />
                </button>
              </div>
              
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={onCreateKb}
                className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 px-4 rounded-xl font-medium flex items-center justify-center space-x-2 hover:shadow-lg transition-all duration-200"
              >
                <Plus size={18} />
                <span>New Knowledge Base</span>
              </motion.button>
            </motion.div>
          ) : (
            <motion.div
              key="collapsed-header"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.2 }}
              className="flex flex-col items-center space-y-3"
            >
              <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center">
                <Sparkles size={18} className="text-white" />
              </div>
              <button
                onClick={onCreateKb}
                className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl flex items-center justify-center hover:shadow-lg transition-all duration-200"
              >
                <Plus size={18} />
              </button>
              <button
                onClick={onToggleCollapse}
                className="p-2 rounded-lg hover:bg-slate-100 transition-colors"
              >
                <ChevronRight size={18} className="text-slate-600" />
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Search Bar */}
      <AnimatePresence>
        {!isCollapsed && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="p-4 border-b border-slate-200"
          >
            <div className="relative">
              <Search size={18} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                placeholder="Search knowledge bases..."
                value={searchTerm}
                onChange={(e) => onSearchChange(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-lg text-sm placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Knowledge Bases List */}
      <div className="flex-1 overflow-hidden">
        <AnimatePresence mode="wait">
          {!isCollapsed ? (
            <motion.div
              key="expanded-list"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="h-full overflow-y-auto scrollbar-thin p-4"
            >
              {isLoading ? (
                <KnowledgeBaseSkeletons />
              ) : (
                <div className="space-y-2">
                  {filteredKbs.length === 0 ? (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="text-center py-12"
                    >
                      <Database size={48} className="mx-auto text-slate-300 mb-3" />
                      <p className="text-slate-500 font-medium">
                        {searchTerm ? 'No matching knowledge bases' : 'No knowledge bases yet'}
                      </p>
                      <p className="text-slate-400 text-sm mt-1">
                        {searchTerm ? 'Try a different search term' : 'Create your first one to get started'}
                      </p>
                    </motion.div>
                  ) : (
                    filteredKbs.map((kb, index) => (
                      <motion.div
                        key={kb.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.05 }}
                      >
                        <KnowledgeBaseItem
                          kb={kb}
                          isActive={selectedKbId === kb.id}
                          onClick={() => onSelectKb(kb.id)}
                        />
                      </motion.div>
                    ))
                  )}
                </div>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="collapsed-list"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="h-full overflow-y-auto p-2 space-y-2"
            >
              {filteredKbs.slice(0, 6).map((kb) => (
                <motion.button
                  key={kb.id}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => onSelectKb(kb.id)}
                  className={`
                    w-12 h-12 rounded-xl flex items-center justify-center text-lg transition-all duration-200
                    ${selectedKbId === kb.id 
                      ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg' 
                      : 'bg-slate-100 hover:bg-slate-200 text-slate-600'
                    }
                  `}
                  title={kb.name}
                >
                  {kb.icon}
                </motion.button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};