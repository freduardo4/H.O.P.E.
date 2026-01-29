import React from 'react';
import Card, { CardContent, CardHeader, CardFooter } from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import StarRating from '@/components/ui/StarRating';
import { fetchMarketplaceListings } from '@/lib/api';

interface Listing {
    id: string;
    title: string;
    description: string;
    price: number;
    version: string;
    compatibility: string;
    rating?: number;
    reviewCount?: number;
}

const MOCK_LISTINGS: Listing[] = [
    {
        id: '1',
        title: 'Stage 1 ECU Tune',
        description: 'Optimized timing and fuel maps for increased performance while maintaining reliability.',
        price: 199.99,
        version: '1.2.0',
        compatibility: 'MED17.5 / Bosch',
        rating: 4.8,
    },
    {
        id: '2',
        title: 'Transmission Shift Logic',
        description: 'Faster shifts and improved response times for DSG/DCT transmissions.',
        price: 149.50,
        version: '1.0.1',
        compatibility: 'DQ250',
        rating: 4.5,
    },
    {
        id: '3',
        title: 'Eco-Performance Hybrid',
        description: 'Best of both worlds: improved economy at cruise and performance under load.',
        price: 175.00,
        version: '2.0.0',
        compatibility: 'EDC17C64',
        rating: 4.2,
    }
];

export default async function MarketplacePage() {
    const apiListings = await fetchMarketplaceListings();
    const listings = (apiListings && apiListings.length > 0 ? apiListings : MOCK_LISTINGS) as Listing[];

    return (
        <div className="p-8 md:p-24 max-w-7xl mx-auto">
            <header className="mb-16">
                <div className="inline-block px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-bold uppercase tracking-wider mb-4">
                    Active Listings
                </div>
                <h1 className="text-5xl font-black text-white mb-4 tracking-tight">
                    Calibration <span className="gradient-text">Marketplace</span>
                </h1>
                <p className="text-gray-400 text-lg max-w-2xl leading-relaxed">
                    Access high-performance ECU calibrations verified by professional engineers. All files are hardware-locked and cryptographically signed.
                </p>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {listings.map((listing: Listing) => (
                    <Card key={listing.id} className="flex flex-col h-full bg-white/5 backdrop-blur-sm">
                        <CardHeader className="flex justify-between items-center bg-white/5">
                            <div className="flex flex-col gap-2">
                                <span className="text-[10px] font-black uppercase tracking-widest text-primary px-2 py-1 rounded bg-primary/10 w-fit">
                                    {listing.compatibility}
                                </span>
                                <div className="flex items-center gap-2">
                                    <StarRating rating={listing.rating || 0} />
                                    {listing.reviewCount !== undefined && (
                                        <span className="text-[10px] text-gray-500 font-mono">
                                            ({listing.reviewCount})
                                        </span>
                                    )}
                                </div>
                            </div>
                            <span className="text-xl font-black text-secondary">
                                ${listing.price.toFixed(2)}
                            </span>
                        </CardHeader>

                        <CardContent className="flex-1">
                            <h2 className="text-xl font-bold mb-3 text-white group-hover:text-primary transition-colors">
                                {listing.title}
                            </h2>
                            <p className="text-gray-400 text-sm leading-relaxed line-clamp-3">
                                {listing.description}
                            </p>
                        </CardContent>

                        <CardFooter className="flex justify-between items-center bg-black/40">
                            <span className="text-xs font-mono text-gray-500">v{listing.version}</span>
                            <Button size="sm">Details</Button>
                        </CardFooter>
                    </Card>
                ))}
            </div>
        </div>
    );
}
