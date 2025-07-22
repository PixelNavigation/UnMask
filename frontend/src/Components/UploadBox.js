"use client";
import styles from "./UploadBox.module.css";
import React, { useRef, useState } from "react";

export default function UploadBox() {
  const [dragActive, setDragActive] = useState(false);
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
    if (files[0]) alert(`Dropped file: ${files[0].name}`);
  };
  const handleClick = () => {
    fileInputRef.current.click();
  };
  const handleChange = e => {
    const file = e.target.files[0];
    if (file) alert(`Selected file: ${file.name}`);
  };

  return (
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
      />
      <div className={styles["upload-box-content"]}>
        <svg width="48" height="48" fill="none" viewBox="0 0 48 48" className={styles["upload-box-icon"]}>
          <rect width="48" height="48" rx="12" fill="#e3eefd" />
          <path d="M24 14v20M24 14l-6 6M24 14l6 6" stroke="#0070f3" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        <p className={styles["upload-box-title"]}>Drag & drop a file here</p>
        <p className={styles["upload-box-desc"]}>
          or <span className={styles["upload-box-browse"]}>click to browse</span>
        </p>
      </div>
    </div>
  );
}
