import type { Metadata } from 'next';
import { DM_Serif_Text, Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

const dmSerifText = DM_Serif_Text({
  weight: '400',
  subsets: ['latin'],
  variable: '--font-dm-serif-text',
});

export const metadata: Metadata = {
  title: 'the-curator',
  description: 'Turning memories into mastery',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang='en'>
      <body
        className={`${inter.className} ${dmSerifText.variable} wrap-anywhere antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
