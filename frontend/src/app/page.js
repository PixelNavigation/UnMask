

import styles from "./page.module.css";
import UploadBox from "../Components/UploadBox";

export default function Home() {
  return (
    <div className={styles.container}>
      <div className="Header">
        <h1>UnMask</h1>
      </div>
      <UploadBox />
    </div>
  );
}
