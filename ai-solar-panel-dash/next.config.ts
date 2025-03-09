import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  
  // API proxy configuration to avoid CORS issues with Flask backend
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.FLASK_API_URL || 'http://localhost:5000'}/:path*`,
      },
    ];
  },
  
  // Image configuration for Firebase Storage images
  images: {
    domains: [
      'firebasestorage.googleapis.com',
      'storage.googleapis.com',
    ],
    formats: ['image/webp'],
  },
  
  // Environment variables exposed to the client
  env: {
    NEXT_PUBLIC_FLASK_API_URL: process.env.FLASK_API_URL || 'http://localhost:5000',
  },
  
  // Increase timeout for API requests to accommodate longer operations
  experimental: {
    serverActions: {
      bodySizeLimit: '10mb',
    },
  },
};

export default nextConfig;
