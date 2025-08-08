"use client";
import styles from "./UploadBox.module.css";
import React, { useRef, useState } from "react";

export default function UploadBox() {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
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

    const formData = new FormData();
    formData.append('video', file);

    try {
      // Connect to the Flask API running on port 5000
      const response = await fetch('http://localhost:5000/api/detect', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setResult(data);
      } else {
        const errorData = await response.json();
        setResult({ error: errorData.error || 'Detection failed' });
      }
    } catch (error) {
      console.error('Backend error:', error);
      setResult({ 
        error: 'Backend connection failed. Please ensure the Flask server is running on port 5000.' 
      });
    } finally {
      setUploading(false);
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
                </div>
                
                {result.faces_detected && (
                  <div className={styles.details}>
                    <h4>📊 Analysis Details</h4>
                    <p><strong>Faces Detected:</strong> {result.faces_detected}</p>
                    <p><strong>Processing Time:</strong> {result.processing_time}s</p>
                    {result.models_used && (
                      <p><strong>Models Used:</strong> {result.models_used.join(', ')}</p>
                    )}
                  </div>
                )}
                
                {result.individual_predictions && (
                  <div className={styles.modelResults}>
                    <h4>🤖 Individual Model Results</h4>
                    {Object.entries(result.individual_predictions).map(([model, pred]) => (
                      <div key={model} className={styles.modelResult}>
                        <span className={styles.modelName}>{model}:</span>
                        <span className={pred.is_fake ? styles.fake : styles.real}>
                          {pred.is_fake ? 'Fake' : 'Real'} ({(pred.confidence * 100).toFixed(1)}%)
                        </span>
                      </div>
                    ))}
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
