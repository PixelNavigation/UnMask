import styles from "./page.module.css";

export default function Home() {
  return (
    <div className={styles.container}>
      <div className="Header">
        <h1>UnMask</h1>
      </div>
      <div className="Search-Box">
        <input type="text" placeholder="Paste the link to check for authenticity" />
      </div>
    </div>
  );
}
