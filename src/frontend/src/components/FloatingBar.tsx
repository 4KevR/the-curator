import TextInput from '@/components/TextInput';
import React, { useRef, useState } from 'react';
import { Download, Mic, MoreVertical, RefreshCw, Upload } from 'react-feather';

interface FloatingBarProps {
  onStartVoiceRecording: () => void;
  onSendTextMessage: (message: string) => void;
  onResetConversation?: () => void;
  onExportAnkiCollection: () => void;
  onImportAnkiCollection: (file: File) => void;
  isRecording: boolean;
}

const FloatingBar: React.FC<FloatingBarProps> = ({
  onStartVoiceRecording,
  onSendTextMessage,
  onResetConversation,
  onExportAnkiCollection,
  onImportAnkiCollection,
  isRecording,
}) => {
  const [isShowingMoreOptions, setIsShowingMoreOptions] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  let blurTimeout: NodeJS.Timeout;
  const wrapperRef = useRef<HTMLDivElement>(null);

  const handleBlur = () => {
    blurTimeout = setTimeout(() => setIsShowingMoreOptions(false), 100);
  };

  const handleFocus = () => {
    if (blurTimeout) clearTimeout(blurTimeout);
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onImportAnkiCollection(file);
    }
  };

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className='fixed right-0 bottom-0 left-0 z-10 flex items-end justify-center space-x-4 p-4'>
      <button
        onClick={onStartVoiceRecording}
        className={`flex cursor-pointer items-center justify-center rounded-full bg-white p-4 text-gray-800 shadow-lg transition-all duration-300 ${isRecording ? 'w-20' : 'w-14 hover:bg-gray-100 focus:ring-2 focus:ring-gray-300 focus:outline-none'}`}
      >
        {isRecording && (
          <span className='me-2 inline-flex h-2 w-2 animate-ping rounded-full bg-yellow-500'></span>
        )}
        <Mic size={24} />
      </button>
      <TextInput onSendText={onSendTextMessage} />
      <div
        className='flex flex-col space-y-2'
        onFocus={handleFocus}
        onBlur={handleBlur}
        tabIndex={-1}
        ref={wrapperRef}
      >
        <button
          onClick={onResetConversation}
          className={`flex cursor-pointer items-center justify-center rounded-full bg-white p-4 text-gray-800 shadow-lg transition-all duration-150 hover:bg-gray-100 focus:ring-2 focus:ring-gray-300 focus:outline-none ${isShowingMoreOptions ? 'pointer-events-auto opacity-100' : 'pointer-events-none opacity-0'}`}
        >
          <RefreshCw size={24} />
        </button>
        <button
          onClick={onExportAnkiCollection}
          className={`flex cursor-pointer items-center justify-center rounded-full bg-white p-4 text-gray-800 shadow-lg transition-all duration-150 hover:bg-gray-100 focus:ring-2 focus:ring-gray-300 focus:outline-none ${isShowingMoreOptions ? 'pointer-events-auto opacity-100' : 'pointer-events-none opacity-0'}`}
        >
          <Upload size={24} />
        </button>
        <button
          onClick={handleImportClick}
          className={`flex cursor-pointer items-center justify-center rounded-full bg-white p-4 text-gray-800 shadow-lg transition-all duration-150 hover:bg-gray-100 focus:ring-2 focus:ring-gray-300 focus:outline-none ${isShowingMoreOptions ? 'pointer-events-auto opacity-100' : 'pointer-events-none opacity-0'}`}
        >
          <Download size={24} />
        </button>
        <button
          className='flex cursor-pointer items-center justify-center rounded-full bg-white p-4 text-gray-800 shadow-lg hover:bg-gray-100 focus:ring-2 focus:ring-gray-300 focus:outline-none'
          onClick={() => setIsShowingMoreOptions(!isShowingMoreOptions)}
        >
          <MoreVertical size={24} />
        </button>
      </div>
      <input
        type='file'
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: 'none' }}
        accept='.apkg'
      />
    </div>
  );
};

export default FloatingBar;
