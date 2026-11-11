import "./globals.css";

export const metadata = {
  title: "ProfilePilot AI",
  description: "Your professional profile. Your personal AI assistant."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
