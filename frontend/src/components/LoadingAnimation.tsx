"use client";

import { useState, useEffect } from "react";

interface LoadingAnimationProps {
  stage: string;
  isVisible: boolean;
}

export default function LoadingAnimation({ stage, isVisible }: LoadingAnimationProps) {
  const [dots, setDots] = useState("");

  useEffect(() => {
    if (!isVisible) return;
    
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? "" : prev + ".");
    }, 400);

    return () => clearInterval(interval);
  }, [isVisible]);

  if (!isVisible) return null;

  const getStageEmoji = (stage: string) => {
    switch (stage) {
      case 'parsing': return '🔍';
      case 'filtering': return '🎯';  
      case 'analysis': return '🧠';
      case 'validation': return '✅';
      default: return '⚡';
    }
  };

  return (
    <div className="fixed inset-0 bg-black/20 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-8 shadow-2xl text-center max-w-md mx-4">
        <div className="text-6xl mb-4 animate-bounce">
          {getStageEmoji(stage)}
        </div>
        <div className="text-xl font-bold mb-2">
          AI is processing{dots}
        </div>
        <div className="text-gray-600 capitalize">
          {stage.replace('_', ' ')} in progress
        </div>
        <div className="mt-4 flex justify-center space-x-2">
          {[...Array(3)].map((_, i) => (
            <div
              key={i}
              className={`w-3 h-3 bg-blue-500 rounded-full animate-pulse`}
              style={{ 
                animationDelay: `${i * 0.2}s`,
                animationDuration: '1s'
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}