'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Bot, User, Loader2 } from 'lucide-react';
import { ChatMessage as ChatMessageType } from '@/lib/types';

interface ChatMessageProps {
  message: ChatMessageType;
  index: number;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, index }) => {
  const isUser = message.type === 'user';
  const isLoading = message.isLoading;

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: true 
    });
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ 
        delay: index * 0.05,
        duration: 0.3,
        ease: 'easeOut'
      }}
      className={`flex items-start space-x-3 mb-6 ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}
    >
      {/* Avatar */}
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ delay: index * 0.05 + 0.1, duration: 0.2 }}
        className={`
          w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg
          ${isUser 
            ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white' 
            : 'bg-gradient-to-r from-emerald-500 to-teal-600 text-white'
          }
        `}
      >
        {isUser ? <User size={18} /> : <Bot size={18} />}
      </motion.div>

      {/* Message Content */}
      <div className={`max-w-2xl ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: index * 0.05 + 0.2, duration: 0.3 }}
          className={`
            relative px-4 py-3 rounded-2xl shadow-sm
            ${isUser 
              ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white' 
              : 'bg-white border border-slate-200 text-slate-800'
            }
          `}
        >
          {isLoading ? (
            <div className="flex items-center space-x-2">
              <Loader2 size={16} className="animate-spin" />
              <span className="text-sm">Thinking...</span>
            </div>
          ) : (
            <div className="whitespace-pre-wrap text-sm leading-relaxed">
              {message.content.split('\n').map((line, i) => {
                if (line.startsWith('•')) {
                  return (
                    <div key={i} className="flex items-start space-x-2 my-1">
                      <span className={`${isUser ? 'text-white' : 'text-blue-500'} font-bold`}>•</span>
                      <span>{line.substring(1).trim()}</span>
                    </div>
                  );
                }
                if (line.includes('**') && line.includes('**')) {
                  const parts = line.split(/\*\*(.*?)\*\*/g);
                  return (
                    <p key={i} className={i > 0 ? 'mt-2' : ''}>
                      {parts.map((part, j) => 
                        j % 2 === 1 ? (
                          <strong key={j} className={isUser ? 'text-white' : 'text-slate-900'}>
                            {part}
                          </strong>
                        ) : (
                          part
                        )
                      )}
                    </p>
                  );
                }
                return line ? <p key={i} className={i > 0 ? 'mt-2' : ''}>{line}</p> : <br key={i} />;
              })}
            </div>
          )}

          {/* Message tail */}
          <div className={`
            absolute top-3 w-3 h-3 rotate-45
            ${isUser 
              ? 'right-[-6px] bg-gradient-to-r from-blue-500 to-purple-600' 
              : 'left-[-6px] bg-white border-l border-b border-slate-200'
            }
          `} />
        </motion.div>

        {/* Timestamp */}
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: index * 0.05 + 0.4, duration: 0.2 }}
          className={`
            text-xs text-slate-500 mt-1 px-2
            ${isUser ? 'text-right' : 'text-left'}
          `}
        >
          {formatTime(message.timestamp)}
        </motion.span>
      </div>
    </motion.div>
  );
};