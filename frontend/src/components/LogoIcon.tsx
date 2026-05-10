export function LogoIcon({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="em-logo-grad" x1="0" y1="0" x2="28" y2="28" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#8b84ff" />
          <stop offset="100%" stopColor="#60a5fa" />
        </linearGradient>
      </defs>
      {/* Face circle */}
      <circle cx="14" cy="14" r="12.25" stroke="url(#em-logo-grad)" strokeWidth="1.5" fill="none" />
      {/* Left eye */}
      <circle cx="10.5" cy="11.5" r="1.5" fill="url(#em-logo-grad)" />
      {/* Right eye */}
      <circle cx="17.5" cy="11.5" r="1.5" fill="url(#em-logo-grad)" />
      {/* Smile */}
      <path d="M10 16.5 Q14 20.5 18 16.5" stroke="url(#em-logo-grad)" strokeWidth="1.75" strokeLinecap="round" fill="none" />
      {/* Subtle glow dots above eyebrows — emotion "sparks" */}
      <circle cx="10.5" cy="8" r="0.75" fill="#8b84ff" opacity="0.55" />
      <circle cx="17.5" cy="8" r="0.75" fill="#60a5fa" opacity="0.55" />
    </svg>
  );
}
