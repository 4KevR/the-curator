import {
  Dialog,
  DialogBackdrop,
  DialogPanel,
  DialogTitle,
  Listbox,
  ListboxButton,
  ListboxOption,
  ListboxOptions,
  Transition,
} from '@headlessui/react';
import { Fragment, useEffect, useState } from 'react';
import { Check, ChevronDown } from 'react-feather';

interface DeckSelectionDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectDeck: (deckName: string) => void;
  userId: string;
}

const DeckSelectionDialog: React.FC<DeckSelectionDialogProps> = ({
  isOpen,
  onClose,
  onSelectDeck,
  userId,
}) => {
  const [decks, setDecks] = useState<string[]>([]);
  const [selectedDeck, setSelectedDeck] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setIsLoading(true);
      setError(null);
      fetch(`http://localhost:5000/api/anki/decks/${userId}`)
        .then((response) => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          return response.json();
        })
        .then((data) => {
          setDecks(data.decks);
          if (data.decks.length > 0) {
            setSelectedDeck(data.decks[0]); // Select the first deck by default
          } else {
            setSelectedDeck(null);
          }
        })
        .catch((err) => {
          console.error('Failed to fetch decks:', err);
          setError('Failed to load decks. Please try again.');
        })
        .finally(() => {
          setIsLoading(false);
        });
    }
  }, [isOpen, userId]);

  const handleExport = () => {
    if (selectedDeck) {
      onSelectDeck(selectedDeck);
      onClose();
    }
  };

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as='div' className='relative z-10' onClose={onClose}>
        <DialogBackdrop
          transition
          className='fixed inset-0 bg-black/30 duration-300 ease-out data-closed:opacity-0'
        />

        <div className='fixed inset-0 overflow-y-auto'>
          <div className='flex min-h-full items-center justify-center p-4 text-center'>
            <DialogPanel
              transition
              className='w-full max-w-md transform rounded-2xl bg-white p-6 text-left align-middle shadow-xl transition-all duration-300 ease-out data-closed:scale-95 data-closed:opacity-0'
            >
              <DialogTitle
                as='h3'
                className='text-lg leading-6 font-medium text-gray-900'
              >
                Select Deck to Export
              </DialogTitle>
              <div className='mt-2'>
                {isLoading ? (
                  <p>Loading decks...</p>
                ) : error ? (
                  <p className='text-red-500'>{error}</p>
                ) : decks.length === 0 ? (
                  <p>No decks found for this user.</p>
                ) : (
                  <div className='relative mt-4'>
                    <Listbox value={selectedDeck} onChange={setSelectedDeck}>
                      <div className='relative mt-1'>
                        <ListboxButton className='focus-visible:ring-opacity-75 relative w-full cursor-default rounded-lg bg-white py-2 pr-10 pl-3 text-left shadow-md focus:outline-none focus-visible:border-indigo-500 focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-offset-2 focus-visible:ring-offset-blue-300 sm:text-sm'>
                          <span className='block truncate'>{selectedDeck}</span>
                          <span className='pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2'>
                            <ChevronDown
                              className='h-5 w-5 text-gray-400'
                              aria-hidden='true'
                            />
                          </span>
                        </ListboxButton>
                        <Transition
                          as={Fragment}
                          leave='transition ease-in duration-100'
                          leaveFrom='opacity-100'
                          leaveTo='opacity-0'
                        >
                          <ListboxOptions className='ring-opacity-5 absolute mt-1 max-h-60 w-full overflow-auto rounded-md bg-white py-1 text-base shadow-lg ring-1 ring-black focus:outline-none sm:text-sm'>
                            {decks.map((deck, deckIdx) => (
                              <ListboxOption
                                key={deckIdx}
                                className='relative cursor-default py-2 pr-4 pl-10 select-none data-[focus]:bg-blue-100 data-[focus]:text-blue-900'
                                value={deck}
                              >
                                {({ selected }) => (
                                  <>
                                    <span
                                      className={`block truncate ${
                                        selected ? 'font-medium' : 'font-normal'
                                      }`}
                                    >
                                      {deck}
                                    </span>
                                    {selected ? (
                                      <span className='absolute inset-y-0 left-0 flex items-center pl-3 text-blue-600'>
                                        <Check
                                          className='h-5 w-5'
                                          aria-hidden='true'
                                        />
                                      </span>
                                    ) : null}
                                  </>
                                )}
                              </ListboxOption>
                            ))}
                          </ListboxOptions>
                        </Transition>
                      </div>
                    </Listbox>
                  </div>
                )}
              </div>

              <div className='mt-4 flex justify-end space-x-2'>
                <button
                  type='button'
                  className='inline-flex justify-center rounded-md border border-transparent bg-blue-100 px-4 py-2 text-sm font-medium text-blue-900 hover:bg-blue-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2'
                  onClick={handleExport}
                  disabled={!selectedDeck || isLoading}
                >
                  Export
                </button>
                <button
                  type='button'
                  className='inline-flex justify-center rounded-md border border-transparent bg-gray-100 px-4 py-2 text-sm font-medium text-gray-900 hover:bg-gray-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-gray-500 focus-visible:ring-offset-2'
                  onClick={onClose}
                >
                  Cancel
                </button>
              </div>
            </DialogPanel>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
};

export default DeckSelectionDialog;
