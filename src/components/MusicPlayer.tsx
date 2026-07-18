import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, Volume2, VolumeX, SkipForward, Upload, Disc } from 'lucide-react';

interface Track {
  name: string;
  artist: string;
  url: string;
  isCustom?: boolean;
}

const PRESET_TRACKS: Track[] = [
  {
    name: 'Lofi Cyber Dreams',
    artist: 'Chill Synthwave',
    url: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3'
  },
  {
    name: 'Retro Coding Beats',
    artist: 'Keyboard Echoes',
    url: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3'
  },
  {
    name: 'Ambient Workspace',
    artist: 'Cosmic Drift',
    url: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3'
  }
];

export default function MusicPlayer() {
  const [tracks, setTracks] = useState<Track[]>(PRESET_TRACKS);
  const [currentTrackIndex, setCurrentTrackIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.5);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const currentTrack = tracks[currentTrackIndex];

  // Sync volume
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = isMuted ? 0 : volume;
    }
  }, [volume, isMuted]);

  // Handle play/pause toggle
  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().then(() => {
        setIsPlaying(true);
      }).catch(err => {
        console.error('Audio play blocked:', err);
      });
    }
  };

  // Skip track
  const skipTrack = () => {
    let nextIndex = (currentTrackIndex + 1) % tracks.length;
    setCurrentTrackIndex(nextIndex);
    setIsPlaying(false);
    setCurrentTime(0);
    // Auto-play the next track after source swap
    setTimeout(() => {
      if (audioRef.current) {
        audioRef.current.load();
        audioRef.current.play().then(() => {
          setIsPlaying(true);
        }).catch(err => console.log('Autoplay skipped', err));
      }
    }, 150);
  };

  // Handle custom audio file upload
  const handleAudioUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const fileUrl = URL.createObjectURL(file);
    const newTrack: Track = {
      name: file.name.replace(/\.[^/.]+$/, ""), // remove extension
      artist: 'Uploaded Local Track',
      url: fileUrl,
      isCustom: true
    };

    setTracks(prev => [...prev, newTrack]);
    setCurrentTrackIndex(tracks.length); // switch to the newly uploaded track
    setIsPlaying(false);
    setCurrentTime(0);

    setTimeout(() => {
      if (audioRef.current) {
        audioRef.current.load();
        audioRef.current.play().then(() => {
          setIsPlaying(true);
        }).catch(err => console.log('Autoplay skipped', err));
      }
    }, 150);
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
    }
  };

  const handleAudioEnded = () => {
    skipTrack();
  };

  // Seek audio
  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const seekVal = parseFloat(e.target.value);
    if (audioRef.current) {
      audioRef.current.currentTime = seekVal;
      setCurrentTime(seekVal);
    }
  };

  const formatTime = (time: number) => {
    if (isNaN(time)) return '0:00';
    const mins = Math.floor(time / 60);
    const secs = Math.floor(time % 60);
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  return (
    <div className="music-player glass-panel">
      {/* Hidden audio element */}
      <audio
        ref={audioRef}
        src={currentTrack.url}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={handleAudioEnded}
      />
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleAudioUpload}
        accept="audio/*"
        style={{ display: 'none' }}
      />

      <div className="player-header">
        <div className="track-info">
          <Disc className={`track-disc-icon ${isPlaying ? 'spinning' : ''}`} size={32} />
          <div className="track-text">
            <span className="track-name">{currentTrack.name}</span>
            <span className="track-artist">{currentTrack.artist}</span>
          </div>
        </div>
        <button
          onClick={() => fileInputRef.current?.click()}
          className="upload-music-btn"
          title="Upload your own song"
        >
          <Upload size={16} />
          <span>Upload MP3</span>
        </button>
      </div>

      {/* Visualizer bars */}
      <div className="visualizer-container">
        <div className={`visualizer-bars ${isPlaying ? 'active' : ''}`}>
          {Array.from({ length: 18 }).map((_, i) => (
            <div
              key={i}
              className="v-bar"
              style={{
                animationDelay: `${i * 0.08}s`,
                height: isPlaying ? undefined : '4px'
              }}
            />
          ))}
        </div>
      </div>

      {/* Progress tracker */}
      <div className="timeline-container">
        <span className="time-text">{formatTime(currentTime)}</span>
        <input
          type="range"
          min={0}
          max={duration || 100}
          value={currentTime}
          onChange={handleSeek}
          className="timeline-slider"
        />
        <span className="time-text">{formatTime(duration)}</span>
      </div>

      {/* Controls */}
      <div className="player-controls">
        <button onClick={skipTrack} className="control-btn" title="Next Track">
          <SkipForward size={20} />
        </button>

        <button onClick={togglePlay} className="play-btn" title={isPlaying ? 'Pause' : 'Play'}>
          {isPlaying ? <Pause size={24} fill="currentColor" /> : <Play size={24} fill="currentColor" style={{ marginLeft: '2px' }} />}
        </button>

        <div className="volume-control">
          <button onClick={() => setIsMuted(!isMuted)} className="control-btn">
            {isMuted || volume === 0 ? <VolumeX size={18} /> : <Volume2 size={18} />}
          </button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={volume}
            onChange={(e) => {
              setVolume(parseFloat(e.target.value));
              setIsMuted(false);
            }}
            className="volume-slider"
          />
        </div>
      </div>
    </div>
  );
}
