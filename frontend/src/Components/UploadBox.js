"use client";
import styles from "./UploadBox.module.css";
import React, { useRef, useState } from "react";

export default function UploadBox() {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [gradcamImage, setGradcamImage] = useState(null);
  const [loadingExplanation, setLoadingExplanation] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = e => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = e => {
    setDragActive(false);
  };

  const handleDrop = e => {
    e.preventDefault();
    setDragActive(false);
    const files = e.dataTransfer.files;
    if (files[0]) {
      uploadFile(files[0]);
    }
  };

  const handleClick = () => {
    fileInputRef.current.click();
  };

  const handleChange = e => {
    const file = e.target.files[0];
    if (file) {
      uploadFile(file);
    }
  };

  const uploadFile = async (file) => {
    setUploading(true);
    setResult(null);
    setExplanation(null);
    setGradcamImage(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Determine endpoint based on file type
      const isVideo = file.type.startsWith('video/');
      const endpoint = isVideo ? 'analyze/video' : 'analyze/image';
      
      // Connect to the Flask API running on port 8000
      const response = await fetch(`http://localhost:8000/${endpoint}`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        // Transform the response to match the frontend's expected format
        const transformedResult = {
          is_deepfake: data.label === 'fake',
          confidence: data.label === 'fake' ? data.confidence : (1 - data.confidence), // How confident we are in the prediction
          fake_probability: data.confidence, // Raw fake probability from model
          faces_detected: data.faces_detected || 1,
          processing_time: data.processing_time || 'N/A',
          models_used: [`${data.details}`],
          file_type: isVideo ? 'video' : 'image',
          fake_frames: data.fake_frames || [], // Frames detected as fake
          gradcam_image: data.gradcam_image || null // Base64 encoded Grad-CAM image
        };
        setResult(transformedResult);
        
        if (data.gradcam_image) {
          setGradcamImage(data.gradcam_image);
        }
        
        // Get AI explanation after getting the result
        generateExplanation(transformedResult);
      } else {
        const errorData = await response.json();
        setResult({ error: errorData.error || 'Detection failed' });
      }
    } catch (error) {
      console.error('Backend error:', error);
      setResult({ 
        error: 'Backend connection failed. Please ensure the Flask server is running on port 8000.' 
      });
    } finally {
      setUploading(false);
    }
  };

  const generateExplanation = async (analysisResult) => {
    setLoadingExplanation(true);
    try {
      const prompt = `Based on this deepfake analysis result, provide a clear explanation of why this ${analysisResult.file_type} was classified as ${analysisResult.is_deepfake ? 'fake' : 'real'}:

- Status: ${analysisResult.is_deepfake ? 'DEEPFAKE DETECTED' : 'APPEARS AUTHENTIC'}
- Confidence: ${(analysisResult.confidence * 100).toFixed(1)}%
- Fake Probability: ${(analysisResult.fake_probability * 100).toFixed(1)}%
- File Type: ${analysisResult.file_type}
${analysisResult.fake_frames.length > 0 ? `- Suspicious Frames: ${analysisResult.fake_frames.join(', ')}` : ''}

Please explain in simple terms what factors likely contributed to this classification and what users should look for when detecting deepfakes.`;

      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.NEXT_PUBLIC_OPENROUTER_API_KEY}`,
          'HTTP-Referer': window.location.origin,
          'X-Title': 'UnMask Deepfake Detector',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'meta-llama/llama-3.2-3b-instruct:free',
          messages: [
            {
              role: 'system',
              content: 'You are an AI expert in deepfake detection. Provide clear, educational explanations about deepfake analysis results in 2-3 sentences.'
            },
            {
              role: 'user',
              content: prompt
            }
          ],
          max_tokens: 150,
          temperature: 0.7,
        }),
      });

      console.log('OpenRouter response status:', response.status);
      
      if (response.ok) {
        const data = await response.json();
        console.log('OpenRouter full response:', JSON.stringify(data, null, 2));
        
        if (data.choices && Array.isArray(data.choices) && data.choices.length > 0) {
          const firstChoice = data.choices[0];
          console.log('First choice:', firstChoice);
          
          if (firstChoice.message && firstChoice.message.content) {
            const content = firstChoice.message.content.trim();
            console.log('Extracted content:', content);
            setExplanation(content);
          } else {
            console.error('Missing message content in choice:', firstChoice);
            setExplanation('AI explanation received but content was empty.');
          }
        } else {
          console.error('No choices in response:', data);
          setExplanation('AI explanation service returned no results.');
        }
      } else {
        const errorData = await response.text();
        console.error('OpenRouter error:', response.status, errorData);
        setExplanation(`Unable to generate AI explanation (Error ${response.status}). Please check your API key.`);
      }
    } catch (error) {
      console.error('Explanation generation error:', error);
      setExplanation('Unable to generate AI explanation at this time.');
    } finally {
      setLoadingExplanation(false);
    }
  };

  return (
    <div className={styles.container}>
      <div
        className={dragActive ? `${styles["upload-box"]} ${styles["dragover"]}` : styles["upload-box"]}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          id="fileInput"
          type="file"
          style={{ display: "none" }}
          onChange={handleChange}
          accept="video/*,image/*"
        />
        <div className={styles["upload-box-content"]}>
          {uploading ? (
            <div className={styles.loading}>
              <div className={styles.spinner}></div>
              <p>Analyzing file...</p>
            </div>
          ) : (
            <>
              <svg width="48" height="48" fill="none" viewBox="0 0 48 48" className={styles["upload-box-icon"]}>
                <rect width="48" height="48" rx="12" fill="#e3eefd" />
                <path d="M24 14v20M24 14l-6 6M24 14l6 6" stroke="#0070f3" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <p className={styles["upload-box-title"]}>Drag & drop a file here</p>
              <p className={styles["upload-box-desc"]}>
                or <span className={styles["upload-box-browse"]}>click to browse</span>
              </p>
            </>
          )}
        </div>
      </div>

      {result && (
        <div className={styles.result}>
          {result.error ? (
            <div className={styles.error}>
              <h3>❌ Error</h3>
              <p>{result.error}</p>
            </div>
          ) : (
            <div className={styles.success}>
              <h3>🎯 Detection Result</h3>
              <div className={styles.resultGrid}>
                <div className={styles.mainResult}>
                  <p><strong>Status:</strong> 
                    <span className={result.is_deepfake ? styles.fake : styles.real}>
                      {result.is_deepfake ? '🚨 DEEPFAKE DETECTED' : '✅ APPEARS AUTHENTIC'}
                    </span>
                  </p>
                  <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%</p>
                  <p><strong>Fake Probability:</strong> {(result.fake_probability * 100).toFixed(1)}%</p>
                  <p><strong>File Type:</strong> {result.file_type}</p>
                </div>
                
                

                {/* Frame Analysis for Videos */}
                {result.file_type === 'video' && result.fake_frames && result.fake_frames.length > 0 && (
                  <div className={styles.frameAnalysis}>
                    <h4>🎬 Suspicious Frames Detected</h4>
                    <p>The following frames were flagged as potentially manipulated:</p>
                    <div className={styles.frameList}>
                      {result.fake_frames.map((frame, index) => (
                        <div key={index} className={styles.frameItem}>
                          Frame {frame}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Grad-CAM Visualization with AI Explanation */}
                {gradcamImage && (
                  <div className={styles.gradcam}>
                    <h4>🔍 Visual Explanation (Grad-CAM)</h4>
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '32px' }}>
                      <div>
                        <img 
                          src={`data:image/png;base64,${gradcamImage}`} 
                          alt="Grad-CAM visualization" 
                          className={styles.gradcamImage}
                        />
                        <p style={{ marginTop: '8px', color: '#92400e', fontWeight: 500 }}>
                          <span style={{ fontWeight: 600 }}>Red areas</span> show regions the AI focused on when making its decision.
                        </p>
                      </div>
                      <div style={{ flex: 1 }}>
                        <h4 style={{ margin: '0 0 8px 0', color: '#0c4a6e', fontSize: '1.05rem', fontWeight: 600 }}>🤖 AI Explanation</h4>
                        {loadingExplanation ? (
                          <div className={styles.loadingExplain}>
                            <div className={styles.loadingSpinner}></div>
                            <span>Generating explanation...</span>
                          </div>
                        ) : explanation ? (
                          <div style={{ background: '#f0f9ff', padding: '12px 16px', borderRadius: '8px', border: '1px solid #0ea5e9', color: '#164e63', fontSize: '0.98rem', fontWeight: 500, lineHeight: 1.7 }}>
                            {explanation.split(/\n|\r/).map((line, idx) => (
                              <p key={idx} style={{ margin: '8px 0' }}>{line.trim()}</p>
                            ))}
                          </div>
                        ) : (
                          <p>AI explanation will appear here after analysis.</p>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
