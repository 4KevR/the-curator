import React, { useEffect, useRef } from 'react';
import { FileText } from 'react-feather';

interface ConversationMessage {
  text: string;
  sender: 'bot' | 'human';
}

interface ConversationHistoryProps {
  messages: ConversationMessage[];
  liveTranscription: string;
  currentState?: string;
  pipelineActive: boolean;
  isRecording?: boolean;
}

const ConversationHistory: React.FC<ConversationHistoryProps> = ({
  messages,
  liveTranscription,
  currentState,
  pipelineActive,
  isRecording = false,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, liveTranscription, isRecording]);

  return (
    <div className='flex-1 p-4'>
      {messages.length === 0 && (
        <div className='flex h-full flex-col items-center justify-center text-xl text-gray-500'>
          <FileText size={48} className='mb-4' />
          <span>Start your conversation by recording or typing a message!</span>
        </div>
      )}
      {messages.length !== 0 && (
        <div className='space-y-2'>
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex ${msg.sender === 'human' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`${msg.sender === 'human' ? 'rounded-l-lg rounded-tr-lg bg-blue-500 text-white' : 'rounded-tl-lg rounded-r-lg bg-gray-300 text-gray-800'} max-w-xs p-3`}
              >
                {msg.text}
              </div>
            </div>
          ))}
          {isRecording === true && (
            <div className='flex justify-end'>
              <div
                className={`rounded-l-lg rounded-tr-lg bg-blue-500 p-3 text-white italic transition-all duration-300`}
                style={{
                  minWidth: '4rem',
                  width: `${Math.min(20, Math.max(4, (liveTranscription?.length || 8) * 0.6 + 4))}rem`,
                  maxWidth: '20rem',
                }}
              >
                {liveTranscription || 'Listening...'}
              </div>
            </div>
          )}
          {pipelineActive === true && (
            <div className='flex justify-start'>
              <div className='text-gray-500 italic'>
                {currentState || 'Waiting for state...'}
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      )}
    </div>
  );
};

export default ConversationHistory;
