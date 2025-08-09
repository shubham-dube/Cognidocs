import React from 'react';

interface LoadingSkeletonProps {
  className?: string;
}

export const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({ className = '' }) => {
  return (
    <div className={`animate-pulse bg-slate-200 rounded-lg ${className}`} />
  );
};

export const KnowledgeBaseSkeletons: React.FC = () => {
  return (
    <div className="space-y-3">
      {[...Array(4)].map((_, i) => (
        <div key={i} className="flex items-center space-x-3 p-3">
          <LoadingSkeleton className="w-10 h-10 rounded-lg" />
          <div className="flex-1">
            <LoadingSkeleton className="h-4 w-3/4 mb-2" />
            <LoadingSkeleton className="h-3 w-1/2" />
          </div>
        </div>
      ))}
    </div>
  );
};

export const ChatMessageSkeletons: React.FC = () => {
  return (
    <div className="space-y-4 p-4">
      {[...Array(3)].map((_, i) => (
        <div key={i} className={`flex ${i % 2 === 0 ? 'justify-end' : 'justify-start'}`}>
          <div className={`max-w-md ${i % 2 === 0 ? 'bg-blue-100' : 'bg-white'} rounded-2xl p-4`}>
            <LoadingSkeleton className="h-4 w-full mb-2" />
            <LoadingSkeleton className="h-4 w-3/4" />
          </div>
        </div>
      ))}
    </div>
  );
};