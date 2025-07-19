import React, { useRef, useState } from 'react';
import { Send, Type } from 'react-feather';

interface TextInputProps {
  onSendText: (message: string) => void;
}

const TextInput: React.FC<TextInputProps> = ({ onSendText }) => {
  const [text, setText] = useState('');
  const [textExpanded, setTextExpanded] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);
  let blurTimeout: NodeJS.Timeout;

  const handleSend = () => {
    if (text.trim()) {
      onSendText(text);
      setText('');
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      handleSend();
    }
  };

  const handleBlur = () => {
    blurTimeout = setTimeout(() => setTextExpanded(false), 100);
  };

  const handleFocus = () => {
    if (blurTimeout) clearTimeout(blurTimeout);
  };

  return (
    <div
      className={`flex h-14 items-center justify-center overflow-hidden rounded-full bg-white shadow-lg transition-all duration-150 ${textExpanded ? 'w-full lg:w-80' : 'w-14 hover:bg-gray-100 focus:outline-none'}`}
      tabIndex={-1}
      ref={wrapperRef}
      onBlur={handleBlur}
      onFocus={handleFocus}
    >
      {textExpanded && (
        <div className='flex w-full items-center'>
          <input
            type='text'
            placeholder='Type your message...'
            className='w-5/6 bg-transparent px-3 py-1 text-gray-800 focus:outline-none'
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyPress}
            autoFocus
          />
          <div className='flex flex-1' />
          <button
            onClick={handleSend}
            className='me-2 cursor-pointer rounded-full bg-blue-600 p-2 text-white hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:outline-none'
          >
            <Send size={20} />
          </button>
        </div>
      )}
      {!textExpanded && (
        <button
          onClick={() => setTextExpanded(true)}
          className='cursor-pointer p-2'
        >
          <Type size={24} />
        </button>
      )}
    </div>
  );
};

export default TextInput;
