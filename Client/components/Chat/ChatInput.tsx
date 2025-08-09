'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Send, Paperclip, Mic, MicOff } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, disabled = false }) => {
  const [message, setMessage] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage('');
      adjustTextareaHeight();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [message]);

  return (
    <div className="border-t border-slate-200 bg-white p-4">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative bg-slate-50 rounded-2xl border border-slate-200 p-3 focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-transparent transition-all duration-200"
        >
          {/* Input Area */}
          <div className="flex items-end space-x-3">
            <button className="p-2 rounded-lg hover:bg-slate-200 transition-colors text-slate-500 hover:text-slate-700">
              <Paperclip size={20} />
            </button>

            <div className="flex-1">
              <textarea
                ref={textareaRef}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={disabled ? 'Select a knowledge base to start chatting...' : 'Ask anything about your documents...'}
                disabled={disabled}
                className="w-full bg-transparent resize-none text-slate-800 placeholder-slate-400 focus:outline-none text-sm leading-relaxed min-h-[20px] max-h-[120px]"
                rows={1}
              />
            </div>

            <div className="flex items-center space-x-2">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setIsRecording(!isRecording)}
                className={`
                  p-2 rounded-lg transition-colors
                  ${isRecording 
                    ? 'bg-red-500 text-white hover:bg-red-600' 
                    : 'hover:bg-slate-200 text-slate-500 hover:text-slate-700'
                  }
                `}
              >
                {isRecording ? <MicOff size={20} /> : <Mic size={20} />}
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleSend}
                disabled={!message.trim() || disabled}
                className={`
                  p-2 rounded-lg transition-all duration-200
                  ${message.trim() && !disabled
                    ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg hover:shadow-xl' 
                    : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  }
                `}
              >
                <Send size={20} />
              </motion.button>
            </div>
          </div>
        </motion.div>

        {/* Input hints */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="flex items-center justify-between mt-2 px-2 text-xs text-slate-400"
        >
          <span>Press Enter to send, Shift + Enter for new line</span>
          <span>{message.length} / 2000</span>
        </motion.div>
      </div>
    </div>
  );
};