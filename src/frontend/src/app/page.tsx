'use client';

import ConversationHistory from '@/components/ConversationHistory';
import DeckSelectionDialog from '@/components/DeckSelectionDialog';
import FloatingBar from '@/components/FloatingBar';
import Header from '@/components/Header';
import SRSContent from '@/components/SRSContent';
import { arrayBufferToBase64, floatTo16BitPCM } from '@/utils/helper';
import { useEffect, useRef, useState } from 'react';
import io, { Socket } from 'socket.io-client';

export type ConversationMessage = {
  text: string;
  sender: 'bot' | 'human';
};

export default function Home() {
  const [conversation, setConversation] = useState<ConversationMessage[]>([]);
  const [srsData, setSrsData] = useState<string[]>([]);
  const [liveTranscription, setLiveTranscription] = useState<string>('');
  const [pipelineActive, setPipelineActive] = useState<boolean>(false);
  const [currentState, setCurrentState] = useState<string>('');
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isDeckSelectionOpen, setIsDeckSelectionOpen] =
    useState<boolean>(false);
  const [userName, setUserName] = useState<string>('web_user');

  const socketRef = useRef<Socket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;

  useEffect(() => {
    const socket = io(backendUrl);
    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('Connected to Socket.IO server');
    });

    socket.on(
      'action_progress',
      (data: { message: string; is_srs_action: boolean }) => {
        if (data.is_srs_action) {
          setSrsData((prev) => [...prev, data.message]);
        } else {
          setCurrentState(data.message);
        }
      },
    );

    socket.on(
      'action_result',
      (data: { task_finish_message?: string; question_answer?: string }) => {
        if (data.task_finish_message) {
          setConversation((prev) => [
            ...prev,
            { text: data.task_finish_message as string, sender: 'bot' },
          ]);
        }
        if (data.question_answer) {
          setConversation((prev) => [
            ...prev,
            { text: data.question_answer as string, sender: 'bot' },
          ]);
        }
        setPipelineActive(false);
      },
    );

    socket.on(
      'action_single_result',
      (data: { task_finish_message: string }) => {
        setConversation((prev) => [
          ...prev,
          { text: data.task_finish_message, sender: 'bot' },
        ]);
        setPipelineActive(false);
      },
    );

    socket.on('streamed_sentence_part', (data: { part: string }) => {
      setLiveTranscription((prev) => prev + data.part);
    });

    socket.on('received_complete_sentence', (data: { sentence: string }) => {
      setConversation((prev) => [
        ...prev,
        { text: data.sentence, sender: 'human' },
      ]);
      setLiveTranscription('');
      stopRecording();
      setPipelineActive(true);
      setCurrentState('');
    });

    socket.on('action_error', (data: { error: string }) => {
      setConversation((prev) => [
        ...prev,
        { text: `Error: ${data.error}`, sender: 'bot' },
      ]);
      setPipelineActive(false);
    });

    socket.on('acknowledged_stream_start', (data: { user: string }) => {
      console.log(`Audio stream started for user: ${data.user}`);
      setLiveTranscription('');
      startRecording();
      setIsRecording(true);
    });

    socket.on('anki_collection_exported', (data: { file_id: string }) => {
      const downloadUrl = `${backendUrl}/api/anki/download/${data.file_id}`;
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = 'exported_anki_deck.apkg';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    });

    return () => {
      socketRef.current?.disconnect();
    };
  }, []);

  const startRecording = async () => {
    // Ensure any existing recording is stopped before starting a new one
    stopRecording();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioCtx = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioCtx;

      await audioCtx.audioWorklet.addModule('/recorder-worklet.js');

      const source = audioCtx.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioCtx, 'recorder-processor');
      workletNodeRef.current = workletNode;

      const bufferSize = 8000; // 0.5 seconds at 16kHz
      let sampleBuffer = new Float32Array(bufferSize);
      let offset = 0;

      workletNode.port.onmessage = (event) => {
        const input = event.data;

        let inputOffset = 0;
        while (inputOffset < input.length) {
          const remainingSpace = bufferSize - offset;
          const samplesToCopy = Math.min(
            input.length - inputOffset,
            remainingSpace,
          );

          sampleBuffer.set(
            input.subarray(inputOffset, inputOffset + samplesToCopy),
            offset,
          );
          offset += samplesToCopy;
          inputOffset += samplesToCopy;

          if (offset >= bufferSize) {
            const pcmChunk = floatTo16BitPCM(sampleBuffer);
            const b64Pcm = arrayBufferToBase64(pcmChunk);
            const duration = pcmChunk.byteLength / (2 * audioCtx.sampleRate);
            if (socketRef.current) {
              socketRef.current.emit('submit_stream_batch', {
                user: userName,
                b64_pcm: b64Pcm,
                duration: duration,
                transcoding: 'client',
              });
            }

            // Reset for next chunk, handling any leftover from current input
            sampleBuffer = new Float32Array(bufferSize);
            offset = 0;
          }
        }
      };

      source.connect(workletNode);
      workletNode.connect(audioCtx.destination);
    } catch (err) {
      console.error('Error accessing microphone:', err);
      setLiveTranscription('Microphone access denied.');
    }
  };

  const stopRecording = () => {
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setLiveTranscription('');
    setIsRecording(false);
  };

  const sendTextMessage = (message: string) => {
    if (socketRef.current) {
      setConversation((prev) => [...prev, { text: message, sender: 'human' }]);
      socketRef.current.emit('submit_action', {
        user: userName,
        transcription: message,
      });
      setPipelineActive(true);
      setCurrentState('');
    }
  };

  const startVoiceRecording = () => {
    if (socketRef.current) {
      socketRef.current.emit('start_audio_streaming', { user: userName });
    }
  };

  const resetConversation = () => {
    setConversation([]);
    setSrsData([]);
    setLiveTranscription('');
    setPipelineActive(false);
    setCurrentState('');
    if (socketRef.current) {
      socketRef.current.emit('new_conversation', { user: userName });
    }
  };

  const handleExportAnkiCollection = () => {
    setIsDeckSelectionOpen(true);
  };

  const handleDeckSelectedForExport = (deckName: string) => {
    if (socketRef.current) {
      socketRef.current.emit('export_anki_collection', {
        user: userName,
        deck_name: deckName,
      });
    }
  };

  const handleImportAnkiCollection = async (file: File) => {
    if (socketRef.current) {
      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch(`${backendUrl}/api/anki/upload`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.file_id) {
          socketRef.current?.emit('import_anki_collection', {
            user: userName,
            file_id: data.file_id,
          });
        } else {
        }
      } catch (error) {
        console.error('Error uploading Anki file:', error);
      }
    }
  };

  const setEventUserName = (name: string) => {
    setUserName(name);
    resetConversation();
  };

  const handleResetAnkiCollection = () => {
    if (socketRef.current) {
      socketRef.current.emit('reset_anki_collection', { user: userName });
    }
  };

  return (
    <div className='flex h-screen flex-col bg-white'>
      <Header
        userName={userName}
        setUserName={setEventUserName}
        onResetAnkiCollection={handleResetAnkiCollection}
      />

      <main className='mt-16 mb-24 flex h-9/12 flex-1 flex-col lg:flex-row'>
        {/* Conversation History Section */}
        <section className='flex w-full flex-1 flex-col overflow-y-auto p-4 lg:w-1/2'>
          <ConversationHistory
            messages={conversation}
            liveTranscription={liveTranscription}
            currentState={currentState}
            pipelineActive={pipelineActive}
            isRecording={isRecording}
          />
        </section>

        {/* SRS Content Section */}
        <section className='flex w-full flex-1 flex-col overflow-y-auto p-4 lg:w-1/2'>
          <SRSContent srsData={srsData} />
        </section>
      </main>

      {/* Floating Action Buttons */}
      <FloatingBar
        onStartVoiceRecording={startVoiceRecording}
        onAbortVoiceRecording={stopRecording}
        onSendTextMessage={sendTextMessage}
        isRecording={isRecording}
        onResetConversation={resetConversation}
        onExportAnkiCollection={handleExportAnkiCollection}
        onImportAnkiCollection={handleImportAnkiCollection}
      />

      <DeckSelectionDialog
        isOpen={isDeckSelectionOpen}
        onClose={() => setIsDeckSelectionOpen(false)}
        onSelectDeck={handleDeckSelectedForExport}
        userId={userName}
      />
    </div>
  );
}
