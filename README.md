# 🔍 OCR Blur Workspace

An interactive, premium web application built with **React, TypeScript, and Vite** that integrates real-time computer vision and text recognition (OCR) to dynamically redact and blur faces, hands, and text from live camera streams or frozen frames.

---

## ✨ Features

- **📷 Real-Time Camera Control:** Toggle your webcam stream on/off and freeze frames to perform precise redactions.
- **👁️ Face & Hand Detection (MediaPipe):**
  - Real-time face tracking and landmark detection using Google's **MediaPipe Tasks Vision**.
  - Intelligent heuristics like **glasses detection** (based on edge density comparison) and **hand-to-face overlap detection**.
  - Configurable auto-blur options for detected faces.
- **📝 Text Recognition (OCR via Tesseract.js):**
  - Run text recognition on the active camera frame.
  - Interactively select and toggle blur on individual detected words.
- **🖌️ Manual Redaction Tools:**
  - Brush/Draw custom blur boxes directly onto the workspace.
  - Adjust blur intensity (radius) dynamically.
- **🎵 Built-in Ambient Music Player:**
  - Chill synthwave and coding tracks to set the perfect work mood.
  - Audio visualizer synced with playback.
  - Supports uploading custom local MP3 files.
- **🎨 Glassmorphic & Modern UI:** Designed with a futuristic dark mode, smooth transitions, and glowing micro-animations.

---

## 🛠️ Tech Stack

- **Framework:** [React 19](https://react.dev/)
- **Build Tool:** [Vite 8](https://vite.dev/)
- **Programming Language:** [TypeScript](https://www.typescriptlang.org/)
- **Packages/Libraries:**
  - [@mediapipe/tasks-vision](https://www.npmjs.com/package/@mediapipe/tasks-vision) — Computer vision and face/hand landmarking.
  - [tesseract.js](https://tesseract.projectnaptha.com/) — OCR engine for word detection and coordinate extraction.
  - [lucide-react](https://lucide.dev/) — Modern UI icons.
- **Package Manager / Runtime:** [Bun](https://bun.sh/) (or npm / yarn)

---

## 🚀 Getting Started

### Prerequisites

Make sure you have [Bun](https://bun.sh/) (or Node.js) installed on your system.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ilhamkrnwan/ocr_blur.git
   cd ocr_blur
   ```

2. **Install dependencies:**
   ```bash
   bun install
   # or npm install / yarn install
   ```

3. **Start the development server:**
   ```bash
   bun dev
   # or npm run dev
   ```

4. **Open in browser:**
   Navigate to `http://localhost:5173` (or the URL output by Vite) to explore the workspace.

---

## 📂 Project Structure

```
ocr_blur/
├── public/                 # Static assets (icons, SVGs, favicon)
├── src/
│   ├── assets/             # Images and design assets
│   ├── components/
│   │   ├── OcrBlurWorkspace.tsx  # Main workspace layout, MediaPipe/Tesseract logic, canvas rendering
│   │   └── MusicPlayer.tsx       # Interactive audio player component with custom upload support
│   ├── App.tsx             # Main React entry point
│   ├── App.css             # Main styling
│   ├── index.css           # Global CSS variables & Tailwind-like resets
│   └── main.tsx            # React DOM mounting
├── package.json            # Configuration and script definitions
└── tsconfig.json           # TypeScript configuration
```

---

## ⚙️ How It Works (Behind the Scenes)

1. **Glasses Detection Heuristic:** The workspace uses a canvas-based edge density comparison. It calculates the high-frequency color gradient variations in the eye region vs. the forehead. If the eye region has significantly higher edge density, it determines that glasses are present.
2. **Text Redaction Canvas:** When OCR is run, Tesseract.js outputs word boundaries (bounding boxes). These are rendered on an overlay layer. Clicking on a word toggles its `isBlurred` flag, which triggers an SVG/Canvas blur filter on that region.

