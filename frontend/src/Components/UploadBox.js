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
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setResult(data);
      } else {
        setResult({ error: data.error || 'Upload failed' });
      }
    } catch (error) {
      setResult({ error: 'Network error. Please check if backend is running.' });
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
          accept="image/*,video/*"
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
              <h3>Error</h3>
              <p>{result.error}</p>
            </div>
          ) : (
            <div className={styles.success}>
              <h3>Detection Result</h3>
              <p><strong>File:</strong> {result.filename}</p>
              <p><strong>Status:</strong> 
                <span className={result.detection_result.is_deepfake ? styles.fake : styles.real}>
                  {result.detection_result.status.toUpperCase()}
                </span>
              </p>
              <p><strong>Confidence:</strong> {(result.detection_result.confidence * 100).toFixed(2)}%</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
