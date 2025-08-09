'use client';

import React, { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, FileText, Plus, Edit3, Trash2, MoreHorizontal } from 'lucide-react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { ChatMessage as ChatMessageType, KnowledgeBase } from '@/lib/types';
import { ChatMessageSkeletons } from '../ui/LoadingSkeleton';

interface ChatAreaProps {
  selectedKb: KnowledgeBase | null;
  messages: ChatMessageType[];
  isLoading: boolean;
  onSendMessage: (message: string) => void;
  onAddDocuments: () => void;
  onEditKb: () => void;
  onDeleteKb: () => void;
}

export const ChatArea: React.FC<ChatAreaProps> = ({
  selectedKb,
  messages,
  isLoading,
  onSendMessage,
  onAddDocuments,
  onEditKb,
  onDeleteKb,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  if (!selectedKb) {
    return (
      <div className="h-full bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="text-center max-w-md"
        >
          <div className="w-24 h-24 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center mx-auto mb-6">
            <MessageSquare size={40} className="text-white" />
          </div>
          <h2 className="text-2xl font-bold text-slate-800 mb-3">
            Welcome to DocuChat
          </h2>
          <p className="text-slate-600 leading-relaxed mb-6">
            Select a knowledge base from the sidebar or create a new one to start chatting with your documents. 
            Upload PDFs, Word docs, images, and more to build your AI-powered knowledge assistant.
          </p>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex items-center justify-center space-x-4 text-sm text-slate-500"
          >
            <div className="flex items-center space-x-1">
              <FileText size={16} />
              <span>Multiple formats</span>
            </div>
            <div className="w-1 h-1 bg-slate-400 rounded-full" />
            <div className="flex items-center space-x-1">
              <MessageSquare size={16} />
              <span>Contextual chat</span>
            </div>
          </motion.div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="h-full bg-slate-50 flex flex-col">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white border-b border-slate-200 p-4 shadow-sm"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center text-lg">
              {selectedKb.icon}
            </div>
            <div>
              <h2 className="font-bold text-slate-900 text-lg">{selectedKb.name}</h2>
              <p className="text-sm text-slate-500">{selectedKb.documentCount} documents</p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onAddDocuments}
              className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-600 hover:text-slate-700 transition-all"
              title="Add Documents"
            >
              <Plus size={18} />
            </motion.button>
            
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onEditKb}
              className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-600 hover:text-slate-700 transition-all"
              title="Edit Knowledge Base"
            >
              <Edit3 size={18} />
            </motion.button>
            
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onDeleteKb}
              className="p-2 rounded-lg bg-slate-100 hover:bg-red-100 text-slate-600 hover:text-red-600 transition-all"
              title="Delete Knowledge Base"
            >
              <Trash2 size={18} />
            </motion.button>
            
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-600 hover:text-slate-700 transition-all"
              title="More Options"
            >
              <MoreHorizontal size={18} />
            </motion.button>
          </div>
        </div>
      </motion.div>

      {/* Chat Messages */}
      <div 
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-4"
      >
        <AnimatePresence mode="wait">
          {isLoading ? (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <ChatMessageSkeletons />
            </motion.div>
          ) : messages.length === 0 ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="text-center py-16"
            >
              <div className="w-16 h-16 rounded-full bg-gradient-to-r from-emerald-500 to-teal-600 flex items-center justify-center mx-auto mb-4">
                <MessageSquare size={24} className="text-white" />
              </div>
              <h3 className="text-lg font-semibold text-slate-800 mb-2">
                Ready to chat with {selectedKb.name}
              </h3>
              <p className="text-slate-600 max-w-md mx-auto">
                Ask questions about your documents and get contextual answers powered by AI. 
                I can help you find information, summarize content, and more.
              </p>
            </motion.div>
          ) : (
            <motion.div
              key="messages"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="max-w-4xl mx-auto"
            >
              {messages.map((message, index) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                  index={index}
                />
              ))}
            </motion.div>
          )}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      {/* Chat Input */}
      <ChatInput
        onSendMessage={onSendMessage}
        disabled={!selectedKb}
      />
    </div>
  );
};