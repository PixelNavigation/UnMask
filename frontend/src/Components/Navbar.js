'use client';

export default function Navbar() {
    return(
        <nav className="navbar">
            <div className="logo">UnMask</div>
            <ul className="nav-links">
                <li><a href="/">Home</a></li>
                <li><a href="/about">About</a></li>
                <li><a href="/contact">Contact</a></li>
            </ul>
        </nav>
    )
}