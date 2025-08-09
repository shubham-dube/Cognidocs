'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { FileText, Calendar } from 'lucide-react';
import { KnowledgeBase } from '@/lib/types';

interface KnowledgeBaseItemProps {
  kb: KnowledgeBase;
  isActive: boolean;
  onClick: () => void;
}

export const KnowledgeBaseItem: React.FC<KnowledgeBaseItemProps> = ({
  kb,
  isActive,
  onClick,
}) => {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric' 
    });
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ x: 4 }}
      transition={{ duration: 0.2 }}
      className={`
        relative p-3 rounded-xl cursor-pointer group transition-all duration-200
        ${isActive 
          ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg' 
          : 'bg-white hover:bg-slate-50 hover:shadow-md'
        }
      `}
      onClick={onClick}
    >
      <div className="flex items-center space-x-3">
        <div className={`
          w-10 h-10 rounded-lg flex items-center justify-center text-lg font-medium
          ${isActive 
            ? 'bg-white/20 text-white' 
            : 'bg-slate-100 text-slate-600 group-hover:bg-slate-200'
          }
        `}>
          {kb.icon}
        </div>
        
        <div className="flex-1 min-w-0">
          <h3 className={`
            font-semibold text-sm truncate
            ${isActive ? 'text-white' : 'text-slate-900'}
          `}>
            {kb.name}
          </h3>
          
          <div className="flex items-center space-x-3 mt-1">
            <div className={`
              flex items-center space-x-1 text-xs
              ${isActive ? 'text-white/80' : 'text-slate-500'}
            `}>
              <FileText size={12} />
              <span>{kb.documentCount} docs</span>
            </div>
            
            <div className={`
              flex items-center space-x-1 text-xs
              ${isActive ? 'text-white/80' : 'text-slate-500'}
            `}>
              <Calendar size={12} />
              <span>{formatDate(kb.lastUpdated)}</span>
            </div>
          </div>
        </div>
      </div>
      
      {isActive && (
        <motion.div
          layoutId="activeIndicator"
          className="absolute inset-0 rounded-xl border-2 border-white/30"
          initial={false}
          transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
        />
      )}
    </motion.div>
  );
};