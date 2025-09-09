/** @type {import('next').NextConfig} */
const nextConfig = {
	// Static export so we can serve via Flask in Docker
	output: 'export',
	// Disable Next image optimization for static export
	images: { unoptimized: true },
};

export default nextConfig;
