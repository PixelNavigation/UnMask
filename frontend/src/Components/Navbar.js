'use client';
import Link from 'next/link';

export default function Navbar() {
    return(
        <nav className="navbar">
            <div className="logo">UnMask</div>
            <ul className="nav-links">
                <li><Link href="/">Home</Link></li>
                <li><Link href="/about">About</Link></li>
                <li><Link href="/contact">Contact</Link></li>
            </ul>
        </nav>
    )
}