import React from 'react';

interface Listing {
    id: string;
    title: string;
    description: string;
    price: number;
    version: string;
    compatibility: string;
}

async function getListings(): Promise<Listing[]> {
    // Simulating GraphQL fetch
    return [
        {
            id: '1',
            title: 'Stage 1 ECU Tune',
            description: 'Optimized timing and fuel maps for increased performance while maintaining reliability.',
            price: 199.99,
            version: '1.2.0',
            compatibility: 'MED17.5 / Bosch',
        },
        {
            id: '2',
            title: 'Transmission Shift Logic',
            description: 'Faster shifts and improved response times for DSG/DCT transmissions.',
            price: 149.50,
            version: '1.0.1',
            compatibility: 'DQ250',
        },
        {
            id: '3',
            title: 'Eco-Performance Hybrid',
            description: 'Best of both worlds: improved economy at cruise and performance under load.',
            price: 175.00,
            version: '2.0.0',
            compatibility: 'EDC17C64',
        }
    ];
}

export default async function MarketplacePage() {
    const listings = await getListings();

    return (
        <div className="min-h-screen bg-[#1A1A1A] text-white p-8 md:p-24">
            <header className="mb-12 border-b border-gray-800 pb-8">
                <h1 className="text-4xl font-bold text-white mb-2">Calibration Marketplace</h1>
                <p className="text-gray-400">Professional ECU tunings, hardware-locked and verified.</p>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {listings.map((listing) => (
                    <div key={listing.id} className="bg-[#252525] border border-gray-700 rounded-xl overflow-hidden hover:border-[#00AAFF] transition-all group">
                        <div className="p-6">
                            <div className="flex justify-between items-start mb-4">
                                <span className="text-xs font-mono text-[#00AAFF] bg-[#00AAFF20] px-2 py-1 rounded">
                                    {listing.compatibility}
                                </span>
                                <span className="text-xl font-bold text-[#FFAA00]">
                                    ${listing.price.toFixed(2)}
                                </span>
                            </div>

                            <h2 className="text-xl font-semibold mb-3 group-hover:text-[#00AAFF] transition-colors">
                                {listing.title}
                            </h2>

                            <p className="text-gray-400 text-sm mb-6 line-clamp-3">
                                {listing.description}
                            </p>

                            <div className="flex justify-between items-center mt-auto">
                                <span className="text-xs text-gray-500">v{listing.version}</span>
                                <button className="bg-[#00AAFF] hover:bg-[#0088CC] text-white px-4 py-2 rounded-lg text-sm font-semibold transition-colors">
                                    View Details
                                </button>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
