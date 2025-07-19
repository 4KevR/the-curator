import React, { useEffect, useRef } from 'react';

interface SRSContentProps {
  srsData: string[];
}

const SRSContent: React.FC<SRSContentProps> = ({ srsData }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [srsData]);

  return (
    <div className='flex-1 overflow-y-auto rounded-lg bg-gray-100 p-4 shadow-inner'>
      <div className='space-y-4'>
        {srsData.length === 0 ? (
          <p className='text-gray-500'>No SRS actions in this session yet.</p>
        ) : (
          srsData.map((item, index) => (
            <div key={index} className='rounded-lg bg-white p-4 shadow'>
              <p className='text-gray-700'>{item}</p>
            </div>
          ))
        )}
      </div>
      <div ref={messagesEndRef} />
    </div>
  );
};

export default SRSContent;
