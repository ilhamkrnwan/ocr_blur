import React, { useState, useRef, useEffect } from 'react';
import * as Tesseract from 'tesseract.js';
import { FilesetResolver, FaceLandmarker, HandLandmarker } from '@mediapipe/tasks-vision';
import { Camera, VideoOff, Download, Trash2, Eye, EyeOff, LayoutGrid, Brush, Settings, ChevronRight } from 'lucide-react';
import MusicPlayer from './MusicPlayer';

interface OcrWord {
  id: string;
  text: string;
  x: number;
  y: number;
  w: number;
  h: number;
  isBlurred: boolean;
}

interface ManualBox {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

interface DetectedFace {
  id: string;
  bbox: { x: number; y: number; w: number; h: number };
  hasGlasses: boolean;
  handOverlapping: boolean;
  shouldBlur: boolean;
  landmarks: any[];
}

interface DetectedHand {
  id: string;
  bbox: { x: number; y: number; w: number; h: number };
}

// Persist static landmarker instances across component mounts
let faceLandmarkerVideo: FaceLandmarker | null = null;
let handLandmarkerVideo: HandLandmarker | null = null;

// Glasses detection heuristic using canvas edge density comparison
function checkGlasses(video: HTMLVideoElement, landmarks: any[]): boolean {
  const width = video.videoWidth;
  const height = video.videoHeight;

  const p130 = landmarks[130];
  const p359 = landmarks[359];
  const p168 = landmarks[168];
  
  if (!p130 || !p359 || !p168) return false;

  const leftX = p130.x * width;
  const rightX = p359.x * width;
  const bridgeY = p168.y * height;
  
  const eyeRegionW = (rightX - leftX) * 1.1;
  const eyeRegionH = (rightX - leftX) * 0.35;
  const eyeRegionX = leftX - (rightX - leftX) * 0.05;
  const eyeRegionY = bridgeY - eyeRegionH * 0.5;

  const p10 = landmarks[10];
  const p9 = landmarks[9];
  if (!p10 || !p9) return false;

  const foreheadY = p10.y * height;
  const foreheadH = Math.max(10, (p9.y - p10.y) * height * 0.8);
  const foreheadW = eyeRegionW * 0.6;
  const foreheadX = leftX + (rightX - leftX) * 0.2;

  const getEdgeDensity = (rx: number, ry: number, rw: number, rh: number): number => {
    const cx = Math.max(0, Math.min(width - 1, rx));
    const cy = Math.max(0, Math.min(height - 1, ry));
    const cw = Math.max(1, Math.min(width - cx, rw));
    const ch = Math.max(1, Math.min(height - cy, rh));

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = cw;
    tempCanvas.height = ch;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return 0;

    tempCtx.drawImage(video, cx, cy, cw, ch, 0, 0, cw, ch);
    try {
      const imgData = tempCtx.getImageData(0, 0, cw, ch);
      const data = imgData.data;

      let edgePixelCount = 0;
      const threshold = 18;
      
      for (let y = 1; y < ch - 1; y++) {
        for (let x = 1; x < cw - 1; x++) {
          const idx = (y * cw + x) * 4;
          
          const val = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
          const valRight = 0.299 * data[(y * cw + (x + 1)) * 4] + 0.587 * data[(y * cw + (x + 1)) * 4 + 1] + 0.114 * data[(y * cw + (x + 1)) * 4 + 2];
          const valDown = 0.299 * data[((y + 1) * cw + x) * 4] + 0.587 * data[((y + 1) * cw + x) * 4 + 1] + 0.114 * data[((y + 1) * cw + x) * 4 + 2];

          const grad = Math.abs(val - valRight) + Math.abs(val - valDown);
          if (grad > threshold) {
            edgePixelCount++;
          }
        }
      }
      return edgePixelCount / (cw * ch);
    } catch (e) {
      console.error("ImageData edge sampling error", e);
      return 0;
    }
  };

  const eyeDensity = getEdgeDensity(eyeRegionX, eyeRegionY, eyeRegionW, eyeRegionH);
  const foreheadDensity = getEdgeDensity(foreheadX, foreheadY, foreheadW, foreheadH);

  return eyeDensity > foreheadDensity * 1.45 && eyeDensity > 0.035;
}

export default function OcrBlurWorkspace() {
  // Camera active states
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [isFeedFrozen, setIsFeedFrozen] = useState(false);
  const [frozenSrc, setFrozenSrc] = useState<string | null>(null);

  // Floating panel collapse state
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);

  // OCR Status
  const [ocrStatus, setOcrStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [ocrStatusMsg, setOcrStatusMsg] = useState('');
  const [ocrWords, setOcrWords] = useState<OcrWord[]>([]);

  // Face & Hand Status
  const [faceStatus, setFaceStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [faceStatusMsg, setFaceStatusMsg] = useState('');
  const [detectedFaces, setDetectedFaces] = useState<DetectedFace[]>([]);
  const [autoBlurFaces, setAutoBlurFaces] = useState(true);

  // Editor states
  const [manualBoxes, setManualBoxes] = useState<ManualBox[]>([]);
  const [blurRadius, setBlurRadius] = useState(18);
  const [toolMode, setToolMode] = useState<'toggle' | 'draw'>('toggle');
  const [showOverlays, setShowOverlays] = useState(true);

  // Drawing state for custom blur boxes
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState({ x: 0, y: 0 });
  const [currentDragBox, setCurrentDragBox] = useState<ManualBox | null>(null);

  // Crop reference to map coordinates accurately
  const cropRef = useRef({ sx: 0, sy: 0, sw: 0, sh: 0 });

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const requestRef = useRef<number | null>(null);

  // Start/Stop camera on mount/unmount
  useEffect(() => {
    startCamera();

    // Resize listener to keep canvas 100vw/100vh
    const handleResize = () => {
      const canvas = canvasRef.current;
      if (canvas) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (frozenSrc && frozenSrc.startsWith('blob:')) {
        URL.revokeObjectURL(frozenSrc);
      }
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Initialize MediaPipe model
  const loadMediaPipe = async () => {
    if (faceLandmarkerVideo && handLandmarkerVideo) return { face: faceLandmarkerVideo, hand: handLandmarkerVideo };

    setFaceStatusMsg('Mengunduh model AI wajah & tangan (~10MB)...');
    
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
    );

    const face = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numFaces: 5
    });

    const hand = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 4
    });

    faceLandmarkerVideo = face;
    handLandmarkerVideo = hand;

    return { face, hand };
  };

  // Camera Management
  const startCamera = async () => {
    setOcrWords([]);
    setDetectedFaces([]);
    setManualBoxes([]);
    setOcrStatus('idle');
    setOcrStatusMsg('');
    setFaceStatus('loading');
    setFaceStatusMsg('Memuat AI pendeteksi...');

    try {
      await loadMediaPipe();

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current.play();
            
            // Set canvas size to match viewport immediately
            const canvas = canvasRef.current;
            if (canvas) {
              canvas.width = window.innerWidth;
              canvas.height = window.innerHeight;
            }

            setIsCameraActive(true);
            setIsFeedFrozen(false);
            setFaceStatus('success');
            setFaceStatusMsg('Kamera aktif.');
          }
        };
      }
      setCameraStream(stream);
    } catch (err) {
      console.error("Camera access error", err);
      setFaceStatus('error');
      setFaceStatusMsg('Gagal mengakses kamera. Nyalakan izin kamera.');
    }
  };

  const stopCamera = () => {
    setIsCameraActive(false);
    setIsFeedFrozen(false);
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setFaceStatus('idle');
    setFaceStatusMsg('');
  };

  // Live render & AI loop for camera feed
  const processVideoFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.paused || video.ended || isFeedFrozen) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Viewport size
    const cWidth = canvas.width;
    const cHeight = canvas.height;

    // Video original size
    const vWidth = video.videoWidth;
    const vHeight = video.videoHeight;
    if (vWidth === 0 || vHeight === 0) return;

    // Cover calculation: center-crop the video frame onto full-screen canvas
    const vRatio = vWidth / vHeight;
    const cRatio = cWidth / cHeight;
    let sx = 0, sy = 0, sw = vWidth, sh = vHeight;

    if (cRatio > vRatio) {
      sh = vWidth / cRatio;
      sy = (vHeight - sh) / 2;
    } else {
      sw = vHeight * cRatio;
      sx = (vWidth - sw) / 2;
    }

    // Save calculations for coordinate mapping
    cropRef.current = { sx, sy, sw, sh };

    // 1. Draw base video frame cropped
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, cWidth, cHeight);

    // 2. Perform Real-time Face & Hand tracking
    let currentFaces: DetectedFace[] = [];
    let currentHands: DetectedHand[] = [];

    // Helper to map normalized coordinates to canvas pixels
    const mapToCanvas = (rect: { x_norm: number; y_norm: number; w_norm: number; h_norm: number }) => {
      return {
        x: ((rect.x_norm * vWidth) - sx) * (cWidth / sw),
        y: ((rect.y_norm * vHeight) - sy) * (cHeight / sh),
        w: rect.w_norm * vWidth * (cWidth / sw),
        h: rect.h_norm * vHeight * (cHeight / sh)
      };
    };

    if (faceLandmarkerVideo && handLandmarkerVideo) {
      try {
        const timestamp = performance.now();
        const faceResult = faceLandmarkerVideo.detectForVideo(video, timestamp);
        const handResult = handLandmarkerVideo.detectForVideo(video, timestamp);

        // Process Hands Bounding Boxes
        if (handResult && handResult.landmarks) {
          handResult.landmarks.forEach((handLandmarks, i) => {
            let hMinX = 1, hMaxX = 0, hMinY = 1, hMaxY = 0;
            handLandmarks.forEach(pt => {
              if (pt.x < hMinX) hMinX = pt.x;
              if (pt.x > hMaxX) hMaxX = pt.x;
              if (pt.y < hMinY) hMinY = pt.y;
              if (pt.y > hMaxY) hMaxY = pt.y;
            });

            const normBbox = {
              x_norm: hMinX,
              y_norm: hMinY,
              w_norm: hMaxX - hMinX,
              h_norm: hMaxY - hMinY
            };

            currentHands.push({
              id: `hand_${i}`,
              bbox: mapToCanvas(normBbox)
            });
          });
        }

        // Process Faces Bounding Boxes
        if (faceResult && faceResult.faceLandmarks) {
          faceResult.faceLandmarks.forEach((faceLandmarks, i) => {
            let fMinX = 1, fMaxX = 0, fMinY = 1, fMaxY = 0;
            faceLandmarks.forEach(pt => {
              if (pt.x < fMinX) fMinX = pt.x;
              if (pt.x > fMaxX) fMaxX = pt.x;
              if (pt.y < fMinY) fMinY = pt.y;
              if (pt.y > fMaxY) fMaxY = pt.y;
            });

            const normFaceBbox = {
              x_norm: fMinX,
              y_norm: fMinY,
              w_norm: fMaxX - fMinX,
              h_norm: fMaxY - fMinY
            };

            const canvasFaceBbox = mapToCanvas(normFaceBbox);

            // Hand overlap verification using canvas mapped coordinates
            let handOverlapping = false;
            currentHands.forEach(hand => {
              if (
                hand.bbox.x < canvasFaceBbox.x + canvasFaceBbox.w &&
                hand.bbox.x + hand.bbox.w > canvasFaceBbox.x &&
                hand.bbox.y < canvasFaceBbox.y + canvasFaceBbox.h &&
                hand.bbox.y + hand.bbox.h > canvasFaceBbox.y
              ) {
                handOverlapping = true;
              }
            });

            const hasGlasses = checkGlasses(video, faceLandmarks);

            currentFaces.push({
              id: `face_${i}`,
              bbox: canvasFaceBbox,
              hasGlasses,
              handOverlapping,
              shouldBlur: hasGlasses || handOverlapping,
              landmarks: faceLandmarks
            });
          });
        }

        setDetectedFaces(currentFaces);
      } catch (err) {
        console.error("Frame tracking error", err);
      }
    }

    // 3. Render Blurs on Canvas
    const blurredFaceRegions = currentFaces.filter(f => f.shouldBlur && autoBlurFaces);
    
    const regions = [
      ...blurredFaceRegions.map(f => ({ x: f.bbox.x, y: f.bbox.y, w: f.bbox.w, h: f.bbox.h })),
      ...manualBoxes.map(b => ({ x: b.x, y: b.y, w: b.w, h: b.h }))
    ];

    regions.forEach(r => {
      if (r.w <= 0 || r.h <= 0) return;
      ctx.save();
      
      // Bounding box mask
      ctx.beginPath();
      ctx.rect(r.x, r.y, r.w, r.h);
      ctx.clip();

      ctx.filter = `blur(${blurRadius}px)`;

      // Apply blur by drawing the canvas back onto itself filtered
      ctx.drawImage(canvas, 0, 0);
      ctx.restore();
    });
  };

  // Loop request animation frame
  const animate = () => {
    if (isCameraActive && !isFeedFrozen) {
      processVideoFrame();
      requestRef.current = requestAnimationFrame(animate);
    }
  };

  useEffect(() => {
    if (isCameraActive && !isFeedFrozen) {
      requestRef.current = requestAnimationFrame(animate);
    } else {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isCameraActive, isFeedFrozen, blurRadius, autoBlurFaces, manualBoxes]);

  // Capture current live frame and trigger OCR
  const captureSnapshotAndScan = async () => {
    const canvas = canvasRef.current;
    if (!canvas || !videoRef.current) return;

    // Freeze loop
    setIsFeedFrozen(true);
    if (requestRef.current) cancelAnimationFrame(requestRef.current);

    // Save snapshot image URL to run Tesseract
    const snapshotUrl = canvas.toDataURL('image/png');
    setFrozenSrc(snapshotUrl);

    setOcrStatus('loading');
    setOcrStatusMsg('Menganalisis teks (OCR)...');

    try {
      const ocrResult = await Tesseract.recognize(
        snapshotUrl,
        'eng',
        {
          logger: (m) => {
            if (m.status === 'recognizing text') {
              setOcrStatusMsg(`Scan Teks: ${Math.round(m.progress * 100)}%`);
            }
          }
        }
      );

      const words: OcrWord[] = (ocrResult.data as any).words.map((w: any, index: number) => ({
        id: `word_${index}_${Date.now()}`,
        text: w.text,
        x: w.bbox.x0,
        y: w.bbox.y0,
        w: w.bbox.x1 - w.bbox.x0,
        h: w.bbox.y1 - w.bbox.y0,
        isBlurred: false
      }));

      // BBoxes returned by OCR are absolute pixels on the captured snapshot image.
      // Since snapshot matches the canvas size, coordinates map 1:1 perfectly!
      setOcrWords(words);
      setOcrStatus('success');
      setOcrStatusMsg('Teks cuplikan berhasil dipindai.');
    } catch (err) {
      console.error(err);
      setOcrStatus('error');
      setOcrStatusMsg('Gagal memindai teks.');
    }
  };

  const resumeLiveCamera = () => {
    setOcrWords([]);
    setOcrStatus('idle');
    setOcrStatusMsg('');
    setIsFeedFrozen(false);
    setFrozenSrc(null);
  };

  // Click handlers
  const handleWordClick = (wordId: string) => {
    if (toolMode !== 'toggle') return;
    setOcrWords(prev =>
      prev.map(w => {
        if (w.id === wordId) {
          const updated = { ...w, isBlurred: !w.isBlurred };
          return updated;
        }
        return w;
      })
    );
  };

  // Re-draw static frozen canvas with redacted elements
  const redrawStaticCanvas = () => {
    const canvas = canvasRef.current;
    const tempImg = new Image();
    if (!canvas || !frozenSrc) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    tempImg.onload = () => {
      ctx.drawImage(tempImg, 0, 0, canvas.width, canvas.height);

      const blurredOcrRegions = ocrWords.filter(w => w.isBlurred);
      const blurredFaceRegions = detectedFaces.filter(f => f.shouldBlur && autoBlurFaces);

      const regions = [
        ...blurredOcrRegions.map(w => ({ x: w.x, y: w.y, w: w.w, h: w.h })),
        ...blurredFaceRegions.map(f => ({ x: f.bbox.x, y: f.bbox.y, w: f.bbox.w, h: f.bbox.h })),
        ...manualBoxes.map(b => ({ x: b.x, y: b.y, w: b.w, h: b.h }))
      ];

      regions.forEach(r => {
        if (r.w <= 0 || r.h <= 0) return;
        ctx.save();
        ctx.beginPath();
        ctx.rect(r.x, r.y, r.w, r.h);
        ctx.clip();
        ctx.filter = `blur(${blurRadius}px)`;
        ctx.drawImage(canvas, 0, 0);
        ctx.restore();
      });
    };
    tempImg.src = frozenSrc;
  };

  useEffect(() => {
    if (isFeedFrozen && frozenSrc) {
      redrawStaticCanvas();
    }
  }, [ocrWords, detectedFaces, manualBoxes, blurRadius, autoBlurFaces, isFeedFrozen, frozenSrc]);

  const handleFaceClick = (faceId: string) => {
    if (toolMode !== 'toggle') return;
    setDetectedFaces(prev =>
      prev.map(f => (f.id === faceId ? { ...f, shouldBlur: !f.shouldBlur } : f))
    );
  };

  // Canvas coordinates for manual drawings (1:1 with window viewport)
  const getCanvasCoords = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      x: clientX - rect.left,
      y: clientY - rect.top
    };
  };

  // Manual box drawings
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (toolMode !== 'draw') return;
    setIsDrawing(true);
    const coords = getCanvasCoords(e.clientX, e.clientY);
    setDrawStart(coords);
    setCurrentDragBox({
      id: 'drag-temp',
      x: coords.x,
      y: coords.y,
      w: 0,
      h: 0
    });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !currentDragBox) return;
    const coords = getCanvasCoords(e.clientX, e.clientY);
    
    const x = Math.min(drawStart.x, coords.x);
    const y = Math.min(drawStart.y, coords.y);
    const w = Math.abs(coords.x - drawStart.x);
    const h = Math.abs(coords.y - drawStart.y);

    setCurrentDragBox({
      id: 'drag-temp',
      x, y, w, h
    });
  };

  const handleMouseUp = () => {
    if (!isDrawing || !currentDragBox) return;
    setIsDrawing(false);

    if (currentDragBox.w > 3 && currentDragBox.h > 3) {
      setManualBoxes(prev => [
        ...prev,
        {
          ...currentDragBox,
          id: `manual_${Date.now()}`
        }
      ]);
    }
    setCurrentDragBox(null);
  };

  const toggleAllBlur = (blur: boolean) => {
    setOcrWords(prev => prev.map(w => ({ ...w, isBlurred: blur })));
    setDetectedFaces(prev => prev.map(f => ({ ...f, shouldBlur: blur })));
  };

  const clearAllBlurs = () => {
    setOcrWords(prev => prev.map(w => ({ ...w, isBlurred: false })));
    setDetectedFaces(prev => prev.map(f => ({ ...f, shouldBlur: false })));
    setManualBoxes([]);
  };

  const downloadImage = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const link = document.createElement('a');
    link.download = `redacted_snapshot_${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  return (
    <div ref={containerRef} style={{ width: '100vw', height: '100vh', position: 'relative', overflow: 'hidden' }}>
      {/* Hidden live video source */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ display: 'none' }}
      />

      {/* Background canvas cover viewport */}
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        className={`${toolMode === 'draw' ? 'cursor-crosshair' : ''}`}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          zIndex: 1,
          display: 'block',
          backgroundColor: '#07080d'
        }}
      />

      {/* Manual Drag Preview Overlay Box */}
      {isDrawing && currentDragBox && (
        <div
          className="drag-preview-box"
          style={{
            position: 'absolute',
            border: '1.5px dashed var(--accent)',
            backgroundColor: 'rgba(168, 85, 247, 0.1)',
            left: `${currentDragBox.x}px`,
            top: `${currentDragBox.y}px`,
            width: `${currentDragBox.w}px`,
            height: `${currentDragBox.h}px`,
            pointerEvents: 'none',
            zIndex: 15
          }}
        />
      )}

      {/* Face overlay boxes */}
      {showOverlays && !isFeedFrozen && detectedFaces.map((face) => (
        <div
          key={face.id}
          onClick={() => handleFaceClick(face.id)}
          className="face-overlay-box"
          style={{
            position: 'absolute',
            left: `${face.bbox.x}px`,
            top: `${face.bbox.y}px`,
            width: `${face.bbox.w}px`,
            height: `${face.bbox.h}px`,
            cursor: toolMode === 'toggle' ? 'pointer' : 'default',
            border: face.shouldBlur && autoBlurFaces ? '2.5px solid var(--accent-red)' : '2px dashed var(--accent-green)',
            backgroundColor: face.shouldBlur && autoBlurFaces ? 'rgba(239, 68, 68, 0.15)' : 'transparent',
            boxShadow: face.shouldBlur && autoBlurFaces ? '0 0 10px rgba(239, 68, 68, 0.4)' : 'none',
            pointerEvents: toolMode === 'toggle' ? 'auto' : 'none',
            zIndex: 20
          }}
        >
          <div
            className="face-badge"
            style={{
              position: 'absolute',
              top: '-26px',
              left: '-2px',
              backgroundColor: face.shouldBlur && autoBlurFaces ? 'var(--accent-red)' : 'var(--accent-green)',
              color: 'white',
              fontSize: '10px',
              padding: '2px 5px',
              borderRadius: '4px',
              whiteSpace: 'nowrap',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              boxShadow: '0 2px 6px rgba(0,0,0,0.5)',
              pointerEvents: 'none'
            }}
          >
            <span>{face.shouldBlur && autoBlurFaces ? '🔒 Foto kita blurr' : '🔓 Foto kita'}</span>
            {face.hasGlasses && <span title="Kacamata terdeteksi">👓</span>}
            {face.handOverlapping && <span title="Tangan terdeteksi">✋</span>}
          </div>
        </div>
      ))}

      {/* OCR Word overlay boxes (when snapshot frozen) */}
      {showOverlays && isFeedFrozen && ocrStatus === 'success' && toolMode === 'toggle' && (
        <div className="ocr-overlay-container" style={{ position: 'absolute', top: 0, left: 0, width: '100vw', height: '100vh', zIndex: 10, pointerEvents: 'none' }}>
          {ocrWords.map((word) => (
            <div
              key={word.id}
              onClick={() => handleWordClick(word.id)}
              className={`ocr-word-box ${word.isBlurred ? 'blurred' : ''}`}
              title={`Klik untuk ${word.isBlurred ? 'membatalkan blur' : 'memblur'}: "${word.text}"`}
              style={{
                position: 'absolute',
                left: `${word.x}px`,
                top: `${word.y}px`,
                width: `${word.w}px`,
                height: `${word.h}px`,
                cursor: 'pointer',
                pointerEvents: 'auto'
              }}
            />
          ))}
        </div>
      )}

      {/* COLLAPSED PANEL TOGGLE BUTTON */}
      {isPanelCollapsed && (
        <button
          onClick={() => setIsPanelCollapsed(false)}
          className="floating-toggle-btn glass-panel"
          title="Buka Control Panel"
          style={{
            position: 'fixed',
            top: '20px',
            right: '20px',
            zIndex: 100,
            width: '48px',
            height: '48px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: '1px solid var(--accent)',
            boxShadow: '0 0 15px var(--accent-glow)',
            cursor: 'pointer',
            background: 'var(--bg-card)',
            color: 'var(--text-primary)'
          }}
        >
          <Settings className="spinning-slow" size={20} />
        </button>
      )}

      {/* EXPANDED FLOATING DASHBOARD PANEL */}
      {!isPanelCollapsed && (
        <div
          className="floating-dashboard glass-panel"
          style={{
            position: 'fixed',
            top: '20px',
            right: '20px',
            zIndex: 100,
            width: '360px',
            maxHeight: '90vh',
            overflowY: 'auto',
            padding: '16px 20px',
            boxSizing: 'border-box',
            display: 'flex',
            flexDirection: 'column',
            gap: '16px'
          }}
        >
          {/* Dashboard Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border)', paddingBottom: '8px' }}>
            <h3 style={{ margin: 0, fontFamily: 'var(--font-display)', fontSize: '16px', color: 'var(--accent-cyan)' }}>
              CyberBlur Dashboard
            </h3>
            <button
              onClick={() => setIsPanelCollapsed(true)}
              className="control-btn"
              title="Sembunyikan Panel"
              style={{ padding: '4px', borderRadius: '4px' }}
            >
              <ChevronRight size={18} />
            </button>
          </div>

          {/* AI Status notifications */}
          {faceStatus === 'loading' && (
            <div style={{ fontSize: '12px', padding: '6px 12px', background: 'rgba(168, 85, 247, 0.1)', borderRadius: '6px', border: '1px solid var(--border-focus)', color: 'var(--accent)' }}>
              {faceStatusMsg}
            </div>
          )}

          {ocrStatus === 'loading' && (
            <div style={{ fontSize: '12px', padding: '6px 12px', background: 'rgba(6, 182, 212, 0.1)', borderRadius: '6px', border: '1px solid var(--border-focus)', color: 'var(--accent-cyan)' }}>
              {ocrStatusMsg}
            </div>
          )}

          {/* Camera Actions */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {!isCameraActive ? (
              <button onClick={startCamera} className="file-upload-btn-lg" style={{ margin: 0, width: '100%', justifyContent: 'center' }}>
                <Camera size={16} />
                <span>Nyalakan Kamera</span>
              </button>
            ) : (
              <div style={{ display: 'flex', gap: '8px' }}>
                <button onClick={stopCamera} className="quick-btn danger-btn" style={{ flex: '1', justifyContent: 'center' }}>
                  <VideoOff size={16} />
                  <span>Matikan</span>
                </button>

                {!isFeedFrozen ? (
                  <button onClick={captureSnapshotAndScan} className="action-btn primary-btn" style={{ flex: '2', justifyContent: 'center', fontSize: '12px', padding: '6px 10px' }}>
                    <span>Pindai Teks OCR</span>
                  </button>
                ) : (
                  <button onClick={resumeLiveCamera} className="action-btn primary-btn" style={{ flex: '2', justifyContent: 'center', fontSize: '12px', padding: '6px 10px', borderColor: 'var(--accent-cyan)', background: 'rgba(6, 182, 212, 0.15)' }}>
                    <span>Live Video</span>
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Control Settings (Visible only when camera active) */}
          {isCameraActive && (
            <>
              {/* Sliders & Tools */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '8px' }}>
                {/* Blur Radius Slider */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <label htmlFor="blur-radius-slider" style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                    Kekuatan Sensor Blur: {blurRadius}px
                  </label>
                  <input
                    id="blur-radius-slider"
                    type="range"
                    min="5"
                    max="50"
                    value={blurRadius}
                    onChange={(e) => setBlurRadius(parseInt(e.target.value))}
                    className="blur-slider"
                    style={{ width: '100%' }}
                  />
                </div>

                {/* Tool Selector */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Metode Sensor:</span>
                  <div className="tool-selector">
                    <button
                      onClick={() => setToolMode('toggle')}
                      className={`tool-btn ${toolMode === 'toggle' ? 'active' : ''}`}
                      title="Pilih Teks OCR / Wajah"
                    >
                      <LayoutGrid size={14} />
                      <span>Pilih</span>
                    </button>
                    <button
                      onClick={() => setToolMode('draw')}
                      className={`tool-btn ${toolMode === 'draw' ? 'active' : ''}`}
                      title="Gambar Manual Blur"
                    >
                      <Brush size={14} />
                      <span>Gambar</span>
                    </button>
                  </div>
                </div>

                {/* Auto face check */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="checkbox"
                    id="autoblur-faces-check"
                    checked={autoBlurFaces}
                    onChange={(e) => setAutoBlurFaces(e.target.checked)}
                    style={{ cursor: 'pointer', accentColor: 'var(--accent)' }}
                  />
                  <label htmlFor="autoblur-faces-check" style={{ fontSize: '12px', cursor: 'pointer', userSelect: 'none', color: 'var(--text-primary)' }}>
                    Auto Sensor Wajah (Kacamata / Tangan)
                  </label>
                </div>
              </div>

              {/* Quick Actions Panel */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                <button onClick={() => toggleAllBlur(true)} className="quick-btn" style={{ flex: '1', fontSize: '11px', padding: '5px 8px' }}>
                  <EyeOff size={12} />
                  <span>Semua</span>
                </button>
                <button onClick={() => toggleAllBlur(false)} className="quick-btn" style={{ flex: '1', fontSize: '11px', padding: '5px 8px' }}>
                  <Eye size={12} />
                  <span>Buka</span>
                </button>
                <button onClick={() => setShowOverlays(!showOverlays)} className="quick-btn" style={{ flex: '1.2', fontSize: '11px', padding: '5px 8px' }}>
                  {showOverlays ? 'Sembunyikan' : 'Tampilkan'}
                </button>
                <button onClick={clearAllBlurs} className="quick-btn danger-btn" style={{ flex: '1', fontSize: '11px', padding: '5px 8px' }}>
                  <Trash2 size={12} />
                  <span>Reset</span>
                </button>
              </div>

              {/* Download Frame (Visible only when frozen) */}
              {isFeedFrozen && (
                <button onClick={downloadImage} className="quick-btn success-btn" style={{ width: '100%', justifyContent: 'center', fontWeight: 'bold' }}>
                  <Download size={14} />
                  <span>Unduh Cuplikan PNG</span>
                </button>
              )}
            </>
          )}

          {/* Compact Music Player Widget (embedded inside the dashboard) */}
          <div style={{ borderTop: '1px solid var(--border)', paddingTop: '14px', marginTop: '4px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', display: 'block', marginBottom: '8px' }}>
              🎵 Lofi Sound System
            </span>
            <MusicPlayer />
          </div>
        </div>
      )}
    </div>
  );
}
