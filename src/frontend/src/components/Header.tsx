import {
  Dialog,
  DialogBackdrop,
  DialogPanel,
  DialogTitle,
  Popover,
  PopoverButton,
  PopoverPanel,
  Transition,
} from '@headlessui/react';
import React, { Fragment, useState } from 'react';
import { Settings } from 'react-feather';

interface HeaderProps {
  userName: string;
  setUserName: (name: string) => void;
  onResetAnkiCollection: () => void;
}

const Header: React.FC<HeaderProps> = ({
  userName,
  setUserName,
  onResetAnkiCollection,
}) => {
  const [pendingUserName, setPendingUserName] = React.useState(userName);
  const [isResetConfirmOpen, setIsResetConfirmOpen] = useState(false);

  const handleUserNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setPendingUserName(event.target.value);
  };

  const handleUserNameSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    setUserName(pendingUserName);
  };

  const handleResetClick = () => {
    setIsResetConfirmOpen(true);
  };

  const handleConfirmReset = () => {
    onResetAnkiCollection();
    setIsResetConfirmOpen(false);
  };

  return (
    <header className='absolute top-0 right-0 left-0 z-10 flex items-center justify-between p-4'>
      <div className='grow text-center'>
        <h1 className='font-dm-serif text-3xl font-bold text-gray-800'>
          the-curator
        </h1>
      </div>
      <div className='absolute right-4'>
        <Popover className='relative'>
          <PopoverButton className='cursor-pointer rounded-full bg-gray-200 p-2 hover:bg-gray-300 focus:ring-2 focus:ring-gray-400 focus:outline-none'>
            <Settings className='h-6 w-6 text-gray-600' />
          </PopoverButton>

          <Transition
            as={Fragment}
            enter='transition ease-out duration-200'
            enterFrom='opacity-0 translate-y-1'
            enterTo='opacity-100 translate-y-0'
            leave='transition ease-in duration-150'
            leaveFrom='opacity-100 translate-y-0'
            leaveTo='opacity-0 translate-y-1'
          >
            <PopoverPanel
              anchor='bottom end'
              className='absolute right-0 z-10 mt-3 w-80 transform px-4 sm:px-0'
            >
              <div className='overflow-hidden rounded-lg border-1 shadow-lg'>
                <div className='relative bg-white p-4'>
                  <h3 className='text-lg font-medium text-gray-900'>
                    Settings
                  </h3>
                  <form onSubmit={handleUserNameSubmit}>
                    <div className='mt-2 flex flex-row items-center justify-center space-x-4'>
                      <span className='text-sm font-medium text-gray-700'>
                        Username
                      </span>
                      <input
                        id='username-input'
                        type='text'
                        value={pendingUserName}
                        onChange={handleUserNameChange}
                        placeholder='Enter username'
                        className='flex grow rounded-md border-gray-300 text-sm shadow-sm focus:border-blue-500 focus:ring-blue-500'
                      />
                    </div>
                    <button
                      type='submit'
                      className='mt-2 w-full rounded-md bg-blue-500 py-1 text-sm font-semibold text-white shadow-sm hover:bg-blue-600 focus:ring-2 focus:ring-blue-400 focus:ring-offset-2 focus:outline-none'
                    >
                      Change User
                    </button>
                  </form>
                  <button
                    onClick={handleResetClick}
                    className='mt-2 w-full rounded-md bg-red-500 py-1 text-sm font-semibold text-white shadow-sm hover:bg-red-600 focus:ring-2 focus:ring-red-400 focus:ring-offset-2 focus:outline-none'
                  >
                    Reset Anki Collection
                  </button>
                </div>
              </div>
            </PopoverPanel>
          </Transition>
        </Popover>
      </div>

      {/* Reset Confirmation Dialog */}
      <Transition appear show={isResetConfirmOpen} as={Fragment}>
        <Dialog
          as='div'
          className='relative z-50'
          onClose={() => setIsResetConfirmOpen(false)}
        >
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
                  Confirm Anki Collection Reset
                </DialogTitle>
                <div className='mt-2'>
                  <p className='text-sm text-gray-500'>
                    Are you sure you want to reset your Anki collection? This
                    action will delete all notes and decks (except the Default
                    deck) and cannot be undone.
                  </p>
                </div>

                <div className='mt-4 flex justify-end space-x-2'>
                  <button
                    type='button'
                    className='inline-flex justify-center rounded-md border border-transparent bg-red-100 px-4 py-2 text-sm font-medium text-red-900 hover:bg-red-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500 focus-visible:ring-offset-2'
                    onClick={handleConfirmReset}
                  >
                    Reset
                  </button>
                  <button
                    type='button'
                    className='inline-flex justify-center rounded-md border border-transparent bg-gray-100 px-4 py-2 text-sm font-medium text-gray-900 hover:bg-gray-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-gray-500 focus-visible:ring-offset-2'
                    onClick={() => setIsResetConfirmOpen(false)}
                  >
                    Cancel
                  </button>
                </div>
              </DialogPanel>
            </div>
          </div>
        </Dialog>
      </Transition>
    </header>
  );
};

export default Header;
