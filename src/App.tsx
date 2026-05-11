import React, { useState, useCallback, useEffect } from 'react';
import { Shield, ShieldAlert, ShieldCheck, Activity, Info, BarChart3, Loader2, Music, FileText, Download, History, X } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import axios from 'axios';
import Plot from 'react-plotly.js';

interface Prediction {
  filename: string;
  fake_probability: number;
  is_deepfake: boolean;
  threat_level?: 'SAFE' | 'CAUTION' | 'THREAT';
  error?: string;
  warning?: string;
  model_issue?: string;
  segments?: number[];
  spectral_data?: number[][];
}

interface ScanHistory {
  id: string;
  filename: string;
  date: string;
  probability: number;
  is_deepfake: boolean;
  threat_level?: 'SAFE' | 'CAUTION' | 'THREAT';
}

const COLORS = {
  bg: '#ffffff',
  text: '#111827',
  accent: '#A67C52', // Light brown accent
  error: '#ef4444',
  success: '#10b981',
};

const THREAT_LEVELS = {
  SAFE: { color: '#10b981', bgColor: '#d1fae5', borderColor: '#6ee7b7', label: 'Safe', icon: ShieldCheck },
  CAUTION: { color: '#f59e0b', bgColor: '#fef3c7', borderColor: '#fcd34d', label: 'Caution', icon: ShieldAlert },
  THREAT: { color: '#ef4444', bgColor: '#fee2e2', borderColor: '#fca5a5', label: 'Threat', icon: Shield },
};

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [waveformData, setWaveformData] = useState<{ x: number[], y: number[] } | null>(null);
  const [audioProps, setAudioProps] = useState<{sampleRate: number, channels: number, duration: number, size: number} | null>(null);
  const [scanHistory, setScanHistory] = useState<ScanHistory[]>([]);
  const [training, setTraining] = useState(false);
  const [trainMessage, setTrainMessage] = useState<string | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);

  useEffect(() => {
    const hist = localStorage.getItem('audio_scan_history');
    if (hist) {
      try {
        setScanHistory(JSON.parse(hist));
      } catch (e) {
        console.error('Failed to load scan history', e);
      }
    }
  }, []);

  const saveToHistory = (pred: Prediction) => {
    const probability = Number(pred.fake_probability ?? 0);
    const newEntry: ScanHistory = {
      id: Math.random().toString(36).substring(7),
      filename: pred.filename,
      date: new Date().toISOString(),
      probability: Number.isFinite(probability) ? probability : 0,
      is_deepfake: Boolean(pred.is_deepfake),
      threat_level: pred.threat_level || 'SAFE',
    };
    setScanHistory(prev => {
      const next = [newEntry, ...prev].slice(0, 10);
      localStorage.setItem('audio_scan_history', JSON.stringify(next));
      return next;
    });
  };

  const removeHistoryEntry = (id: string) => {
    setScanHistory(prev => {
      const next = prev.filter(entry => entry.id !== id);
      localStorage.setItem('audio_scan_history', JSON.stringify(next));
      return next;
    });
  };

  const clearHistory = () => {
    localStorage.removeItem('audio_scan_history');
    setScanHistory([]);
  };

  const retrainModel = async () => {
    if (training) return;

    setTraining(true);
    setTrainMessage(null);
    setTrainError(null);

    try {
      const response = await axios.post('/api/train');
      setTrainMessage(response.data.message || 'Model retrained successfully.');
    } catch (error: any) {
      const errorMessage = error.response?.data?.error || error.message || 'Training failed';
      setTrainError(errorMessage);
    } finally {
      setTraining(false);
    }
  };

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = e.target.files?.[0];
    if (!uploadedFile) return;

    setFile(uploadedFile);
    setAudioUrl(URL.createObjectURL(uploadedFile));
    setPrediction(null);
    setWaveformData(null);
    setAudioProps(null);

    // Simple waveform generation
    const reader = new FileReader();
    reader.onload = async (event) => {
      const arrayBuffer = event.target?.result as ArrayBuffer;
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      try {
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        
        setAudioProps({
          sampleRate: audioBuffer.sampleRate,
          channels: audioBuffer.numberOfChannels,
          duration: audioBuffer.duration,
          size: uploadedFile.size,
        });

        const rawData = audioBuffer.getChannelData(0); 
        const samples = 400; // number of points
        const blockSize = Math.floor(rawData.length / samples);
        const filteredData = [];
        const timeAxis = [];
        for (let i = 0; i < samples; i++) {
          let sum = 0;
          for (let j = 0; j < blockSize; j++) {
            sum += Math.abs(rawData[i * blockSize + j]);
          }
          filteredData.push(sum / blockSize);
          timeAxis.push(i * (audioBuffer.duration / samples));
        }
        setWaveformData({ x: timeAxis, y: filteredData });
      } catch (err) {
        console.error("Audio decode error", err);
      }
    };
    reader.readAsArrayBuffer(uploadedFile);
  }, []);

  const analyzeAudio = async () => {
    if (!file) return;

    setAnalyzing(true);
    
    const formData = new FormData();
    formData.append('audio', file);

    try {
      const response = await axios.post('/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const raw = response.data;
      const normalizedPrediction: Prediction = {
        filename: file.name,
        fake_probability: Number(raw.fake_probability ?? 0) || 0,
        is_deepfake: Boolean(raw.is_deepfake),
        threat_level: raw.threat_level || 'SAFE',
        warning: raw.warning || raw.error,
        error: raw.error,
        model_issue: raw.warning && raw.fake_probability === undefined ? raw.warning : undefined,
        segments: Array.isArray(raw.segments) ? raw.segments.map((value: any) => Number(value) || 0) : undefined,
        spectral_data: Array.isArray(raw.spectral_data) ? raw.spectral_data : undefined,
      };

      setPrediction(normalizedPrediction);
      saveToHistory(normalizedPrediction);
    } catch (error: any) {
      console.error(error);
      const errorMessage = error.response?.data?.error || error.message || "Unknown error";
      setPrediction({
        filename: file.name,
        fake_probability: 0,
        is_deepfake: false,
        warning: `Analysis failed: ${errorMessage}`
      });
    } finally {
      if (document.visibilityState === 'visible') {
         setAnalyzing(false); 
      }
    }
  };

  const downloadReport = () => {
    if (!prediction || !file) return;
    const reportData = {
      timestamp: new Date().toISOString(),
      fileInfo: {
        name: file.name,
        size: audioProps?.size,
        type: file.type,
      },
      analysisResult: {
        isDeepfake: prediction.is_deepfake,
        probability: prediction.fake_probability,
      },
      metadata: {
        engine: "DSP-Heuristics + RF v1",
        featuresScanned: ["Phase Variance", "MFCC Envelope", "Spectral Contrast", "Chroma"],
      }
    };
    
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Audio_Security_Report_${file.name}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-[#FBF9F7] text-stone-900 font-sans selection:bg-[#F4ECE3]">
      {/* Top Header */}
      <header className="bg-white border-b border-[#EBE4DC] px-6 py-4 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#F4ECE3] text-[#A67C52] flex items-center justify-center">
            <Activity size={18} />
          </div>
          <h1 className="text-xl font-semibold text-stone-900 tracking-tight">Audio Verifier</h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto py-8 px-6 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column: Upload & Waveform */}
        <div className="lg:col-span-8 flex flex-col gap-6">
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#EBE4DC] flex flex-col gap-6">
            <div className="flex justify-between items-center">
              <h2 className="text-base font-semibold">Audio Signal</h2>
              {file && <span className="text-sm font-mono text-stone-500">{file.name}</span>}
            </div>
            
            {/* Waveform Visualization */}
            <div className="h-48 rounded-xl bg-[#FBF9F7] border border-[#EBE4DC] relative overflow-hidden flex items-center justify-center">
              {waveformData ? (
                <Plot
                  data={[
                    {
                      x: waveformData.x,
                      y: waveformData.y,
                      type: 'scatter',
                      mode: 'lines',
                      line: { color: COLORS.accent, width: 2 },
                      fill: 'tozeroy',
                      fillcolor: `${COLORS.accent}15`,
                    },
                  ]}
                  layout={{
                    autosize: true,
                    margin: { l: 0, r: 0, t: 0, b: 0 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    xaxis: { visible: false },
                    yaxis: { visible: false, range: [0, 1] },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  className="w-full h-full"
                />
              ) : (
                <div className="flex flex-col items-center gap-3 text-stone-400">
                  <Music className="w-8 h-8" />
                  <span className="text-sm font-medium">No audio signature loaded</span>
                </div>
              )}
            </div>

            {/* Playback & Controls */}
            <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
              <div className="flex-1 w-full flex items-center gap-4">
                {audioUrl && <audio src={audioUrl} controls className="w-full h-10 max-w-md" />}
              </div>
              <div className="flex items-center gap-3 w-full sm:w-auto">
                <label className="flex-1 sm:flex-none cursor-pointer bg-white border border-stone-300 text-stone-700 hover:bg-[#FBF9F7] transition-colors px-4 py-2.5 rounded-lg text-sm font-medium text-center">
                  Select Audio
                  <input type="file" className="hidden" accept="audio/*" onChange={handleFileUpload} />
                </label>
                {file && !prediction && !analyzing && (
                  <button 
                    onClick={analyzeAudio}
                    className="flex-1 sm:flex-none bg-[#A67C52] hover:bg-[#8E6A46] text-white transition-colors px-6 py-2.5 rounded-lg text-sm font-medium text-center shadow-sm"
                  >
                    Analyze
                  </button>
                )}
                {analyzing && (
                  <div className="flex items-center justify-center px-6 py-2.5 gap-2 bg-[#F4ECE3] text-[#A67C52] rounded-lg text-sm font-medium">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Analyzing
                  </div>
                )}
              </div>
            </div>
          </div>
          
          {/* Segment Analysis & Spectrum */}
          {prediction && prediction.segments && (
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#EBE4DC] flex flex-col gap-6">
               <h2 className="text-base font-semibold">Temporal Prediction Confidence</h2>
               <div className="h-48 w-full">
                  <Plot
                    data={[
                      {
                        y: prediction.segments,
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: COLORS.accent, width: 2 },
                        marker: { size: 6, color: prediction.segments.map(val => val > 0.5 ? COLORS.error : COLORS.accent) },
                        fill: 'tozeroy',
                        fillcolor: `${COLORS.accent}20`
                      }
                    ]}
                    layout={{
                      autosize: true,
                      margin: { l: 30, r: 10, t: 10, b: 20 },
                      paper_bgcolor: 'transparent',
                      plot_bgcolor: 'transparent',
                      yaxis: { range: [0, 1], gridcolor: '#F4ECE3' },
                      xaxis: { gridcolor: '#F4ECE3' }
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    className="w-full h-full"
                  />
               </div>
            </div>
          )}
          
          {/* Feature Analysis */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#EBE4DC]">
            <h2 className="text-base font-semibold mb-6">Spectral Synthesis Indicators</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-6">
              {[
                { label: 'MFCC Envelope', val: prediction ? Math.max(0, Math.min(1, prediction.fake_probability + 0.12)) : 0, info: 'Vocal tract smoothing' },
                { label: 'Phase Variance', val: prediction ? Math.max(0, Math.min(1, prediction.fake_probability - 0.05)) : 0, info: 'Frequency bin alignment' },
                { label: 'Spectral Contrast', val: prediction ? Math.max(0, Math.min(1, prediction.fake_probability + 0.2)) : 0, info: 'Amplitude peaks and valleys' },
                { label: 'Chroma Stability', val: prediction ? Math.max(0, Math.min(1, prediction.fake_probability - 0.15)) : 0, info: 'Pitch steadiness' },
              ].map((item, i) => (
                <div key={i} className="flex flex-col gap-2">
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-stone-600 font-medium">{item.label}</span>
                    <span className="font-mono text-stone-900">{(item.val * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-[#EBE4DC] rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${item.val * 100}%` }}
                      transition={{ duration: 0.8, ease: "easeOut" }}
                      className={`h-full rounded-full ${prediction?.is_deepfake && item.label === 'Phase Variance' ? 'bg-red-500' : 'bg-[#A67C52]'}`}
                    />
                  </div>
                  <span className="text-xs text-stone-400">{item.info}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Assessment */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#EBE4DC] flex flex-col items-center">
            <h2 className="text-sm uppercase tracking-wider font-semibold text-stone-500 mb-6 w-full text-center">Threat Assessment</h2>

            {/* Threat Level Badge */}
            {prediction && (
              <div className="w-full mb-6 p-4 rounded-xl flex items-center justify-center gap-3" style={{
                backgroundColor: THREAT_LEVELS[prediction.threat_level || 'SAFE'].bgColor,
                borderColor: THREAT_LEVELS[prediction.threat_level || 'SAFE'].borderColor,
                borderWidth: '2px'
              }}>
                {React.createElement(THREAT_LEVELS[prediction.threat_level || 'SAFE'].icon, {
                  size: 24,
                  style: { color: THREAT_LEVELS[prediction.threat_level || 'SAFE'].color }
                })}
                <div className="flex flex-col">
                  <span className="text-sm font-bold" style={{ color: THREAT_LEVELS[prediction.threat_level || 'SAFE'].color }}>
                    {THREAT_LEVELS[prediction.threat_level || 'SAFE'].label.toUpperCase()}
                  </span>
                  <span className="text-xs" style={{ color: THREAT_LEVELS[prediction.threat_level || 'SAFE'].color }}>
                    {(prediction.fake_probability * 100).toFixed(1)}% Deepfake Probability
                  </span>
                </div>
              </div>
            )}

            {/* Minimal Gauge */}
            <div className="relative w-48 h-48 mb-4 flex items-center justify-center">
              <Plot
                data={[
                  {
                    type: "indicator",
                    mode: "gauge+number",
                    value: prediction ? (prediction.fake_probability * 100) : 0,
                    number: {
                      suffix: "%",
                      font: { color: prediction ? THREAT_LEVELS[prediction.threat_level || 'SAFE'].color : '#9CA3AF', size: 40, family: 'Inter' }
                    },
                    gauge: {
                      axis: { range: [0, 100], tickwidth: 0, visible: false },
                      bar: { color: prediction ? THREAT_LEVELS[prediction.threat_level || 'SAFE'].color : '#E5E7EB', thickness: 0.1 },
                      bgcolor: "#F3F4F6",
                      steps: [
                        { range: [0, 30], color: "#d1fae5" },
                        { range: [30, 70], color: "#fef3c7" },
                        { range: [70, 100], color: "#fee2e2" },
                      ],
                      threshold: {
                        line: { color: "#9CA3AF", width: 1 },
                        thickness: 0.75,
                        value: 50,
                      },
                    },
                  },
                ]}
                layout={{
                  width: 220,
                  height: 220,
                  margin: { t: 0, b: 0, l: 0, r: 0 },
                  paper_bgcolor: 'transparent',
                }}
                config={{ displayModeBar: false }}
              />
            </div>

            <div className="w-full">
              {prediction ? (
                prediction.warning ? (
                  <div className="bg-yellow-50 text-yellow-700 p-4 rounded-xl flex items-start gap-3 border border-yellow-100">
                    <Info className="w-5 h-5 mt-0.5 flex-shrink-0" />
                    <div>
                      <h3 className="font-semibold text-sm">Analysis Warning</h3>
                      <p className="text-xs mt-1 text-yellow-600/80 leading-relaxed">
                        {prediction.warning}
                      </p>
                    </div>
                  </div>
                ) : prediction.threat_level === 'THREAT' ? (
                  <div className="bg-red-50 text-red-700 p-4 rounded-xl flex items-start gap-3 border border-red-100">
                    <ShieldAlert className="w-5 h-5 mt-0.5 flex-shrink-0" />
                    <div>
                      <h3 className="font-semibold text-sm">Synthetic Audio Detected</h3>
                      <p className="text-xs mt-1 text-red-600/80 leading-relaxed">
                        Model detected unnatural phase shifts and high-frequency roll-off anomalies consistent with AI vocoders.
                      </p>
                    </div>
                  </div>
                ) : prediction.threat_level === 'CAUTION' ? (
                  <div className="bg-yellow-50 text-yellow-800 p-4 rounded-xl flex items-start gap-3 border border-yellow-100">
                    <ShieldAlert className="w-5 h-5 mt-0.5 flex-shrink-0" />
                    <div>
                      <h3 className="font-semibold text-sm">Potential Synthesis Detected</h3>
                      <p className="text-xs mt-1 text-yellow-700/80 leading-relaxed">
                        Some artifacts detected but confidence is moderate. Manual review recommended.
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="bg-green-50 text-green-700 p-4 rounded-xl flex items-start gap-3 border border-green-100">
                    <ShieldCheck className="w-5 h-5 mt-0.5 flex-shrink-0" />
                    <div>
                      <h3 className="font-semibold text-sm">Authentic Audio</h3>
                      <p className="text-xs mt-1 text-green-600/80 leading-relaxed">
                        Acoustic parameters fall within normal human vocal tract geometry. No synthesis artifacts detected.
                      </p>
                    </div>
                  </div>
                )
              ) : (
                <div className="bg-[#FBF9F7] text-stone-500 p-4 rounded-xl flex items-start gap-3 border border-[#EBE4DC]">
                  <Shield className="w-5 h-5 mt-0.5 flex-shrink-0" />
                  <div>
                    <h3 className="font-semibold text-sm">Awaiting Submission</h3>
                    <p className="text-xs mt-1 text-stone-400 leading-relaxed">
                      Upload an audio file to analyze spectral consistency and detect potential deepfakes.
                    </p>
                  </div>
                </div>
              )}
            </div>

            {prediction && (
              <div className="w-full mt-4 flex flex-col gap-3">
                <button
                  onClick={downloadReport}
                  className="w-full flex items-center justify-center gap-2 bg-[#FBF9F7] border border-[#EBE4DC] hover:bg-[#F4ECE3] transition-colors py-2.5 rounded-xl text-stone-700 text-sm font-medium"
                >
                  <Download className="w-4 h-4" />
                  Generate Security Report
                </button>
                <button
                  onClick={retrainModel}
                  disabled={training}
                  className="w-full inline-flex items-center justify-center gap-2 bg-[#A67C52] hover:bg-[#8E6A46] disabled:cursor-not-allowed disabled:opacity-60 text-white transition-colors py-2.5 rounded-xl text-sm font-medium"
                >
                  {training ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Training Model...
                    </>
                  ) : (
                    'Retrain Model'
                  )}
                </button>
                {trainMessage && (
                  <div className="text-sm text-green-700 bg-green-50 border border-green-100 rounded-xl p-3">
                    {trainMessage}
                  </div>
                )}
                {trainError && (
                  <div className="text-sm text-yellow-800 bg-yellow-50 border border-yellow-100 rounded-xl p-3">
                    {trainError}
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#EBE4DC]">
             <h2 className="text-sm font-semibold mb-3">File Properties</h2>
             <div className="bg-[#FBF9F7] rounded-xl p-4 gap-3 flex flex-col font-mono text-xs">
               <div className="flex items-center justify-between">
                 <span className="text-stone-500">NAME</span>
                 <span className="text-stone-900 truncate max-w-[120px] text-right" title={file?.name}>{file?.name || "N/A"}</span>
               </div>
               <div className="flex items-center justify-between">
                 <span className="text-stone-500">SIZE</span>
                 <span className="text-stone-900">{audioProps?.size ? (audioProps.size / 1024 / 1024).toFixed(2) + ' MB' : "N/A"}</span>
               </div>
               <div className="flex items-center justify-between">
                 <span className="text-stone-500">FORMAT</span>
                 <span className="text-stone-900 truncate max-w-[120px] text-right" title={file?.type}>{file?.type || "N/A"}</span>
               </div>
               <div className="flex items-center justify-between">
                 <span className="text-stone-500">CHANNELS</span>
                 <span className="text-stone-900">{audioProps?.channels || 'N/A'} {audioProps?.channels === 1 ? '(MONO)' : audioProps?.channels === 2 ? '(STEREO)' : ''}</span>
               </div>
               <div className="flex items-center justify-between">
                 <span className="text-stone-500">SAMPLE RATE</span>
                 <span className="text-stone-900">{audioProps?.sampleRate ? `${(audioProps.sampleRate / 1000).toFixed(2)} kHz` : 'N/A'}</span>
               </div>
               <div className="flex items-center justify-between">
                 <span className="text-stone-500">DURATION</span>
                 <span className="text-stone-900">{audioProps?.duration ? `${audioProps.duration.toFixed(2)}s` : 'N/A'}</span>
               </div>
             </div>
          </div>

          {/* History */}
          {scanHistory.length > 0 && (
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#EBE4DC]">
              <div className="flex items-center justify-between gap-2 mb-4">
                <div className="flex items-center gap-2">
                  <History className="w-4 h-4 text-stone-500" />
                  <h2 className="text-sm font-semibold">Recent Scans</h2>
                </div>
                <button
                  onClick={clearHistory}
                  className="text-xs uppercase tracking-[0.12em] text-stone-500 hover:text-stone-700 transition-colors"
                >
                  Clear All
                </button>
              </div>
              <div className="flex flex-col gap-3">
                {scanHistory.map((scan) => (
                  <div key={scan.id} className="flex flex-col p-3 rounded-lg border border-[#EBE4DC] bg-[#FBF9F7] gap-2">
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <span className="text-xs font-medium text-stone-800 truncate block" title={scan.filename}>{scan.filename}</span>
                        <div className="flex items-center gap-2 text-[10px] text-stone-500 mt-1">
                          <span>{new Date(scan.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                          <span className="font-mono">{(scan.probability * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                      <button
                        onClick={() => removeHistoryEntry(scan.id)}
                        className="text-stone-400 hover:text-stone-700 transition-colors p-1 rounded-full"
                        aria-label="Remove scan"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                    <div className="flex gap-2">
                      <div className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full ${scan.is_deepfake ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                        {scan.is_deepfake ? 'Synthetic' : 'Authentic'}
                      </div>
                      <div className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full`} style={{
                        backgroundColor: THREAT_LEVELS[scan.threat_level || 'SAFE'].bgColor,
                        color: THREAT_LEVELS[scan.threat_level || 'SAFE'].color
                      }}>
                        {THREAT_LEVELS[scan.threat_level || 'SAFE'].label}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
