# whistx Project Context

## Overview

`whistx` is a browser-based real-time transcription app built around an OpenAI-compatible Whisper ASR API. It captures audio from microphone, screen sharing, or both, and streams transcript segments back to the UI via WebSocket.

## Architecture

- **Backend**: FastAPI (Python 3.10+) with WebSocket support
- **Frontend**: Vanilla JS with `MediaRecorder` API
- **Key Features**: Real-time transcription, speaker diarization, LLM-powered proofreading and summarization

### Key Files

- `web/` - Frontend assets
- `server/` - FastAPI backend
- `server/app.py` - Thin entrypoint
- `server/core/application.py` - App wiring and router registration
- `server/api/routes/*.py` - HTTP entrypoints
- `server/api/ws/transcribe.py` - WebSocket entrypoint

---

## Design Context

### Users

Business professionals who need to transcribe meetings and create meeting minutes. They use the app in office environments, often during multi-participant meetings. The job to be done is capturing accurate transcripts with speaker identification for later reference and documentation.

### Brand Personality

**Modern, Simple, Intellectual**

- Clean and sophisticated without being cold
- Efficient but not utilitarian
- Professional yet approachable
- Trustworthy for handling sensitive business conversations

### Aesthetic Direction

**Minimal & Quiet**

- Generous whitespace, breathing room
- Subtle, restrained color palette
- Typography-first design approach
- Calm, focused atmosphere that doesn't distract from the content
- Light and dark mode support (both equally polished)

**Anti-References:**
- AI-generated looking designs with generic aesthetics
- Overly decorative or flashy interfaces
- Cluttered, information-dense layouts

**References (implicit from codebase):**
- Apple's clean minimalism
- Linear's refined simplicity
- Vercel's technical elegance
- Stripe's professional polish

### Design Principles

1. **Content is King**: The transcript text should be the hero. UI elements should recede and support, never compete.

2. **Quiet Confidence**: Design should feel assured and professional without being loud. Use subtle animations and muted transitions.

3. **Typography-First**: Rely on IBM Plex Sans JP's excellent legibility. Use type hierarchy rather than color to create structure.

4. **Glass, Not Chrome**: Prefer glass-morphism and soft shadows over hard borders and heavy chrome. The interface should feel light and layered.

5. **Accessibility by Default**: Japanese as primary language, support for screen readers, clear focus states, and readable contrast ratios.

### Existing Design Tokens

```css
/* Fonts */
--font-sans: "IBM Plex Sans JP", "Segoe UI", sans-serif;
--font-mono: "IBM Plex Mono", "SF Mono", Consolas, monospace;

/* Accent Color - Business Teal */
--accent-600: #24665d;

/* Light/Dark mode both supported */
```

### UI Patterns

- **Glass panels** with blur backdrop for major sections
- **Pill-shaped buttons** (radius-full)
- **Subtle shadows** with multiple layers
- **Consistent 4px spacing scale**
- **Panel resizers** for user-adjustable layouts
- **Collapsible panels** to reduce visual noise
