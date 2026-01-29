import Button from "@/components/ui/Button";
import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center p-8 md:p-24 text-center">
      {/* Hero Section */}
      <section className="max-w-4xl mx-auto py-20 relative">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-64 h-64 bg-primary/20 blur-[120px] -z-10 rounded-full"></div>
        <div className="absolute bottom-0 right-0 w-48 h-48 bg-accent/20 blur-[100px] -z-10 rounded-full"></div>

        <div className="inline-block px-4 py-1.5 mb-8 rounded-full bg-white/5 border border-white/10 text-primary text-xs font-black uppercase tracking-[0.3em]">
          Engineered for Performance
        </div>

        <h1 className="text-6xl md:text-8xl font-black text-white mb-8 tracking-tighter leading-[0.9]">
          Hardware-Optimized <br />
          <span className="gradient-text tracking-[-0.05em]">Performance Engineering</span>
        </h1>

        <p className="text-gray-400 text-xl md:text-2xl mb-12 max-w-2xl mx-auto leading-relaxed font-medium">
          The ultimate cloud ecosystem for ECU tuners and automotive engineers. Secure calibrations, AI-powered diagnostics, and community-driven repair patterns.
        </p>

        <div className="flex flex-col sm:flex-row gap-6 justify-center">
          <Link href="/marketplace">
            <Button size="lg" className="w-full sm:w-auto px-12 text-lg uppercase tracking-widest shadow-2xl shadow-primary/40">
              Browse Marketplace
            </Button>
          </Link>
          <Link href="/wiki">
            <Button variant="outline" size="lg" className="w-full sm:w-auto px-12 text-lg uppercase tracking-widest bg-white/5">
              Explore Wiki-Fix
            </Button>
          </Link>
        </div>
      </section>

      {/* Feature Highlights */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-7xl mx-auto py-32 border-t border-white/5 mt-20">
        <div className="p-8 text-left glass rounded-3xl hover:border-primary/50 transition-colors">
          <div className="text-primary mb-6">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="M12 16v-4" /><path d="M12 8h.01" /></svg>
          </div>
          <h3 className="text-2xl font-bold text-white mb-4">AI Diagnostics</h3>
          <p className="text-gray-400 leading-relaxed font-medium">LSTM-based anomaly detection and Ghost Curve visualizations for precise mechanical insights.</p>
        </div>

        <div className="p-8 text-left glass rounded-3xl hover:border-primary/50 transition-colors">
          <div className="text-primary mb-6">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v20" /><path d="m17 5-5-3-5 3" /><path d="m17 19-5 3-5-3" /><path d="M2 12h20" /><path d="m5 7-3 5 3 5" /><path d="m19 7 3 5-3 5" /></svg>
          </div>
          <h3 className="text-2xl font-bold text-white mb-4">Secure Exchange</h3>
          <p className="text-gray-400 leading-relaxed font-medium">AES-256 encrypted calibration delivery with full hardware-level fingerprinting and licensing.</p>
        </div>

        <div className="p-8 text-left glass rounded-3xl hover:border-primary/50 transition-colors">
          <div className="text-primary mb-6">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="m16 12-4-4-4 4" /><path d="M12 16V8" /></svg>
          </div>
          <h3 className="text-2xl font-bold text-white mb-4">Knowledge Graph</h3>
          <p className="text-gray-400 leading-relaxed font-medium">Billion-node automotive graph indexing millions of DTC codes and real-world repair patterns.</p>
        </div>
      </section>

      <footer className="py-20 text-gray-600 text-sm font-bold uppercase tracking-widest border-t border-white/5 w-full">
        &copy; 2026 H.O.P.E. Engineering &bull; All Rights Reserved
      </footer>
    </div>
  );
}
