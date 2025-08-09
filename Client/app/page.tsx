'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Sidebar } from '@/components/Sidebar/Sidebar';
import { ChatArea } from '@/components/Chat/ChatArea';
import { CreateKnowledgeBaseModal } from '@/components/Modals/CreateKnowledgeBaseModal';
import { AddDocumentsModal } from '@/components/Modals/AddDocumentsModal';
import { KnowledgeBase, ChatMessage } from '@/lib/types';
import { mockKnowledgeBases, mockChatMessages } from '@/lib/mockData';
import toast from 'react-hot-toast';

export default function Home() {
  // State management
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [selectedKbId, setSelectedKbId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoadingKbs, setIsLoadingKbs] = useState(true);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  
  // Modal states
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [isAddDocsModalOpen, setIsAddDocsModalOpen] = useState(false);

  // Load initial data
  useEffect(() => {
    const loadKnowledgeBases = async () => {
      try {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));
        setKnowledgeBases(mockKnowledgeBases);
        
        // Auto-select first KB for demo
        if (mockKnowledgeBases.length > 0) {
          setSelectedKbId(mockKnowledgeBases[0].id);
        }
      } catch (error) {
        toast.error('Failed to load knowledge bases');
      } finally {
        setIsLoadingKbs(false);
      }
    };

    loadKnowledgeBases();
  }, []);

  // Load messages when KB changes
  useEffect(() => {
    if (selectedKbId) {
      loadMessages(selectedKbId);
    } else {
      setMessages([]);
    }
  }, [selectedKbId]);

  const loadMessages = async (kbId: string) => {
    setIsLoadingMessages(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500));
      setMessages(mockChatMessages[kbId] || []);
    } catch (error) {
      toast.error('Failed to load chat history');
    } finally {
      setIsLoadingMessages(false);
    }
  };

  const handleSelectKb = (kbId: string) => {
    setSelectedKbId(kbId);
  };

  const handleSendMessage = async (content: string) => {
    if (!selectedKbId) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      content,
      type: 'user',
      timestamp: new Date().toISOString(),
    };

    const loadingMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      content: '',
      type: 'ai',
      timestamp: new Date().toISOString(),
      isLoading: true,
    };

    setMessages(prev => [...prev, userMessage, loadingMessage]);

    try {
      // Simulate API call to AI service
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const aiResponse = generateMockResponse(content);
      const aiMessage: ChatMessage = {
        id: loadingMessage.id,
        content: aiResponse,
        type: 'ai',
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => 
        prev.map(msg => msg.id === loadingMessage.id ? aiMessage : msg)
      );
    } catch (error) {
      setMessages(prev => prev.filter(msg => msg.id !== loadingMessage.id));
      toast.error('Failed to send message');
    }
  };

  const generateMockResponse = (userMessage: string): string => {
    const responses = [
      "Based on your documents, I can provide you with detailed information about that topic. Let me analyze the relevant sections and provide a comprehensive answer.",
      "I've found several relevant passages in your uploaded documents that address this question. Here's what I can tell you:\n\n• **Key Point 1**: Important information from your documents\n• **Key Point 2**: Additional context and details\n• **Key Point 3**: Supporting evidence and examples\n\nWould you like me to elaborate on any of these points?",
      "Great question! According to the documents in your knowledge base, here's what you should know:\n\nThis topic is covered extensively in your uploaded materials. The main considerations include policy details, coverage limits, and specific procedures you should follow.\n\nIs there a particular aspect you'd like me to focus on?",
      "I can help you with that! From analyzing your documents, here are the most relevant details:\n\n• **Process Overview**: Step-by-step guidance from your materials\n• **Requirements**: What you need to know or prepare\n• **Timeline**: Expected timeframes and deadlines\n• **Contact Information**: Relevant parties or departments\n\nLet me know if you need clarification on any of these areas."
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
  };

  const handleCreateKb = async (name: string, files: File[]) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const newKb: KnowledgeBase = {
      id: Date.now().toString(),
      name,
      icon: '📁',
      lastUpdated: new Date().toISOString().split('T')[0],
      documentCount: files.length,
      createdAt: new Date().toISOString().split('T')[0],
    };

    setKnowledgeBases(prev => [newKb, ...prev]);
    setSelectedKbId(newKb.id);
  };

  const handleAddDocuments = async (files: File[]) => {
    if (!selectedKbId) return;

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    setKnowledgeBases(prev =>
      prev.map(kb =>
        kb.id === selectedKbId
          ? { ...kb, documentCount: kb.documentCount + files.length, lastUpdated: new Date().toISOString().split('T')[0] }
          : kb
      )
    );
  };

  const handleEditKb = () => {
    toast.success('Edit functionality coming soon!');
  };

  const handleDeleteKb = () => {
    if (!selectedKbId) return;
    
    toast((t) => (
      <div className="flex items-center space-x-3">
        <div>
          <p className="font-medium">Delete Knowledge Base?</p>
          <p className="text-sm text-gray-600">This action cannot be undone</p>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => {
              setKnowledgeBases(prev => prev.filter(kb => kb.id !== selectedKbId));
              setSelectedKbId(null);
              toast.dismiss(t.id);
              toast.success('Knowledge base deleted');
            }}
            className="bg-red-500 text-white px-3 py-1 rounded text-sm"
          >
            Delete
          </button>
          <button
            onClick={() => toast.dismiss(t.id)}
            className="bg-gray-200 text-gray-800 px-3 py-1 rounded text-sm"
          >
            Cancel
          </button>
        </div>
      </div>
    ), { duration: 10000 });
  };

  const selectedKb = knowledgeBases.find(kb => kb.id === selectedKbId) || null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="h-screen bg-slate-50 flex"
    >
      {/* Sidebar */}
      <Sidebar
        knowledgeBases={knowledgeBases}
        isLoading={isLoadingKbs}
        selectedKbId={selectedKbId}
        isCollapsed={isSidebarCollapsed}
        searchTerm={searchTerm}
        onSelectKb={handleSelectKb}
        onToggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
        onSearchChange={setSearchTerm}
        onCreateKb={() => setIsCreateModalOpen(true)}
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        <ChatArea
          selectedKb={selectedKb}
          messages={messages}
          isLoading={isLoadingMessages}
          onSendMessage={handleSendMessage}
          onAddDocuments={() => setIsAddDocsModalOpen(true)}
          onEditKb={handleEditKb}
          onDeleteKb={handleDeleteKb}
        />
      </div>

      {/* Modals */}
      <CreateKnowledgeBaseModal
        isOpen={isCreateModalOpen}
        onClose={() => setIsCreateModalOpen(false)}
        onSubmit={handleCreateKb}
      />

      <AddDocumentsModal
        isOpen={isAddDocsModalOpen}
        knowledgeBase={selectedKb}
        onClose={() => setIsAddDocsModalOpen(false)}
        onSubmit={handleAddDocuments}
      />
    </motion.div>
  );
}