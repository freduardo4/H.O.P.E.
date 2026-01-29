import React from 'react';
import Link from 'next/link';
import Button from '../ui/Button';

const Navbar = () => {
    return (
        <nav className="fixed top-0 left-0 right-0 z-50 glass border-b border-white/5 px-6 py-4">
            <div className="max-w-7xl mx-auto flex justify-between items-center">
                <Link href="/" className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg gradient-bg flex items-center justify-center font-bold text-white shadow-lg shadow-primary/20">
                        H
                    </div>
                    <span className="text-xl font-bold tracking-tight text-white uppercase italic">O.P.E</span>
                    <span className="text-xs text-primary font-bold uppercase tracking-widest ml-1 hidden md:inline">Central</span>
                </Link>

                <div className="hidden md:flex items-center gap-8">
                    <Link href="/marketplace" className="text-sm font-medium text-gray-400 hover:text-primary transition-colors">Marketplace</Link>
                    <Link href="/wiki" className="text-sm font-medium text-gray-400 hover:text-primary transition-colors">Wiki-Fix</Link>
                    <Link href="/fleet" className="text-sm font-medium text-gray-400 hover:text-primary transition-colors opacity-50 cursor-not-allowed">Fleet</Link>
                </div>

                <div className="flex items-center gap-4">
                    <Button variant="ghost" size="sm" className="hidden sm:inline-flex">Sign In</Button>
                    <Button variant="primary" size="sm">Get Started</Button>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
