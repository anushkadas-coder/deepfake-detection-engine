import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { UploadCloud, CheckCircle2, XCircle, Terminal, Activity, FileDigit, Cpu, Network } from 'lucide-react';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loadingLogs, setLoadingLogs] = useState([]);
  const fileInputRef = useRef(null);

  // Simulated terminal logs during loading
  useEffect(() => {
    if (loading) {
      const logs = [
        "Initializing ResNet-18 weights...",
        "Extracting facial landmarks...",
        "Applying 2D Fast Fourier Transform...",
        "Shifting zero-frequency components...",
        "Calculating magnitude spectrum...",
        "Normalizing tensor arrays...",
        "Running forward pass...",
        "Computing final sigmoid activation..."
      ];
      let i = 0;
      setLoadingLogs([logs[0]]);
      const interval = setInterval(() => {
        i++;
        if (i < logs.length) {
          setLoadingLogs(prev => [...prev, logs[i]]);
        }
      }, 400);
      return () => clearInterval(interval);
    } else {
      setLoadingLogs([]);
    }
  }, [loading]);

  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]);
  };
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) processFile(e.target.files[0]);
  };

  const processFile = (file) => {
    if (!file.type.startsWith('image/') && !file.type.startsWith('video/')) {
      setError("ERR: Invalid format. Awaiting image or video file.");
      return;
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const resetEngine = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleScan = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // CHANGED: Pointing to your live Hugging Face Space instead of localhost
      const response = await axios.post('https://anushkadas-deepfake-detection-api.hf.space/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setTimeout(() => {
        setResult(response.data);
        setLoading(false);
      }, 1500);
    } catch (err) {
      // CHANGED: Updated error message for Cloud Deployment
      setError("FATAL: Cloud Inference engine unreachable. Ensure Hugging Face Space is 'Running'.");
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#09090b] text-zinc-100 font-sans selection:bg-indigo-500/30 flex flex-col items-center p-4 sm:p-8 relative">
      
      {/* Subtle Grid Background */}
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-overlay pointer-events-none"></div>
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none"></div>

      {/* Top Navigation / Status Bar */}
      <header className="w-full max-w-5xl flex justify-between items-center mb-12 relative z-10 border-b border-zinc-800/50 pb-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center justify-center shadow-sm">
            <Activity className="w-4 h-4 text-zinc-300" />
          </div>
          <span className="font-semibold tracking-tight text-zinc-100">Inference<span className="text-zinc-500">Engine</span></span>
        </div>
        <div className="flex items-center gap-3">
          <div className="px-2 py-1 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-mono tracking-wider flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
            SYSTEM ONLINE
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="w-full max-w-5xl relative z-10 flex flex-col lg:flex-row gap-6">
        
        {/* Left Column: Input & Controls */}
        <div className="flex-1 flex flex-col gap-6">
          
          <div className="bg-[#18181b] border border-zinc-800 rounded-2xl p-6 shadow-2xl flex flex-col h-full">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
                <FileDigit className="w-4 h-4 text-zinc-400" />
                Input Tensor
              </h2>
              {previewUrl && !loading && !result && (
                <button onClick={resetEngine} className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors">Clear</button>
              )}
            </div>

            {!previewUrl ? (
              <div 
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current.click()}
                className={`flex-1 flex flex-col items-center justify-center rounded-xl border border-dashed transition-all cursor-pointer min-h-[300px]
                  ${isDragging ? 'border-indigo-500 bg-indigo-500/5' : 'border-zinc-700 bg-zinc-900/50 hover:border-zinc-500 hover:bg-zinc-800/50'}`}
              >
                <UploadCloud className="w-8 h-8 text-zinc-500 mb-3" />
                <p className="text-sm font-medium text-zinc-300">Select media for analysis</p>
                <p className="text-xs text-zinc-600 mt-1 font-mono">JPG, PNG, WEBP, MP4, MOV</p>
                <input type="file" className="hidden" accept="image/*,video/*" onChange={handleFileChange} ref={fileInputRef} />
              </div>
            ) : (
              <div className="flex-1 flex flex-col relative rounded-xl overflow-hidden border border-zinc-800 bg-[#09090b]">
                
                {selectedFile && selectedFile.type.startsWith('video/') ? (
                  <video src={previewUrl} autoPlay loop muted playsInline className="w-full h-full object-contain absolute inset-0 opacity-80 mix-blend-screen" />
                ) : (
                  <img src={previewUrl} alt="Target" className="w-full h-full object-contain absolute inset-0 opacity-80 mix-blend-screen" />
                )}
                
                {/* Processing Overlay */}
                {loading && (
                  <div className="absolute inset-0 bg-[#09090b]/90 backdrop-blur-sm z-10 p-6 flex flex-col justify-end font-mono text-xs">
                    <div className="flex items-center gap-2 text-indigo-400 mb-4">
                      <Terminal className="w-4 h-4 animate-pulse" />
                      <span>EXECUTING PIPELINE...</span>
                    </div>
                    <div className="space-y-2 text-zinc-500">
                      {loadingLogs.map((log, idx) => (
                        <p key={idx} className="animate-in fade-in slide-in-from-bottom-2 duration-300">
                          <span className="text-zinc-700">{'>'}</span> {log}
                        </p>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Action Button */}
            <button 
              onClick={previewUrl && !result ? handleScan : resetEngine}
              disabled={loading || (!previewUrl && !result)}
              className={`mt-6 w-full py-3 rounded-xl text-sm font-semibold tracking-wide transition-all shadow-lg flex items-center justify-center gap-2
                ${loading ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed border border-zinc-700' 
                : result ? 'bg-zinc-100 text-zinc-900 hover:bg-white' 
                : previewUrl ? 'bg-indigo-600 text-white hover:bg-indigo-500 shadow-indigo-900/20' 
                : 'bg-zinc-800 text-zinc-500 border border-zinc-700 cursor-not-allowed'}`}
            >
              {loading ? 'ANALYZING...' : result ? 'LOAD NEW TENSOR' : 'INITIALIZE FORWARD PASS'}
            </button>
          </div>
        </div>

        {/* Right Column: Bento Grid Results */}
        <div className="flex-1 flex flex-col gap-4">
          
          {error ? (
            <div className="bg-red-500/10 border border-red-500/20 rounded-2xl p-6 text-red-400 text-sm font-mono flex items-start gap-3">
              <XCircle className="w-5 h-5 shrink-0" />
              {error}
            </div>
          ) : result ? (
            <>
              {/* Top Bento Card: Primary Prediction */}
              <div className={`rounded-2xl p-6 border flex flex-col justify-center animate-in slide-in-from-right-4 duration-500
                ${result.prediction === 'REAL' ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-rose-500/5 border-rose-500/20'}`}>
                <div className="flex items-center gap-2 mb-4">
                  {result.prediction === 'REAL' ? <CheckCircle2 className="w-5 h-5 text-emerald-400" /> : <XCircle className="w-5 h-5 text-rose-400" />}
                  <span className={`text-xs font-mono uppercase tracking-widest ${result.prediction === 'REAL' ? 'text-emerald-500' : 'text-rose-500'}`}>Classification</span>
                </div>
                <h1 className={`text-4xl font-bold tracking-tighter ${result.prediction === 'REAL' ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {result.prediction === 'REAL' ? 'AUTHENTIC_HUMAN' : 'SYNTHETIC_ANOMALY'}
                </h1>
              </div>

              {/* Bottom Row Bento Cards */}
              <div className="grid grid-cols-2 gap-4 animate-in slide-in-from-right-4 duration-700 delay-100">
                
                {/* Confidence Card */}
                <div className="bg-[#18181b] border border-zinc-800 rounded-2xl p-6 flex flex-col justify-between">
                  <span className="text-xs font-mono text-zinc-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                    <Activity className="w-4 h-4" /> Confidence
                  </span>
                  <div>
                    <div className="flex items-end gap-1 mb-2">
                      <span className="text-5xl font-light tracking-tighter text-zinc-100 font-mono">{result.confidence}</span>
                      <span className="text-xl text-zinc-500 font-mono pb-1">%</span>
                    </div>
                    <div className="h-1 w-full bg-zinc-800 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${result.prediction === 'REAL' ? 'bg-emerald-500' : 'bg-rose-500'}`} 
                        style={{ width: `${result.confidence}%` }} 
                      />
                    </div>
                  </div>
                </div>

                {/* Architecture Info Card */}
                <div className="bg-[#18181b] border border-zinc-800 rounded-2xl p-6 flex flex-col justify-between">
                  <span className="text-xs font-mono text-zinc-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                    <Network className="w-4 h-4" /> Pipeline
                  </span>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center border-b border-zinc-800/50 pb-2">
                      <span className="text-xs text-zinc-400">Model</span>
                      <span className="text-xs font-mono text-indigo-400">ResNet-18</span>
                    </div>
                    <div className="flex justify-between items-center border-b border-zinc-800/50 pb-2">
                      <span className="text-xs text-zinc-400">Pre-processing</span>
                      <span className="text-xs font-mono text-indigo-400">2D FFT</span>
                    </div>
                    <div className="flex justify-between items-center pb-1">
                      <span className="text-xs text-zinc-400">Params</span>
                      <span className="text-xs font-mono text-indigo-400">~11.1M</span>
                    </div>
                  </div>
                </div>

              </div>
            </>
          ) : (
            <div className="h-full bg-[#18181b]/50 border border-zinc-800/50 border-dashed rounded-2xl p-6 flex flex-col items-center justify-center text-center">
              <Cpu className="w-10 h-10 text-zinc-800 mb-4" />
              <p className="text-sm text-zinc-400 font-medium">Awaiting Input Tensor</p>
              <p className="text-xs text-zinc-600 mt-2 font-mono max-w-[250px]">
                Upload an image or video to initialize the analysis pipeline and view output metrics.
              </p>
            </div>
          )}

        </div>
      </main>

    </div>
  );
}

export default App;