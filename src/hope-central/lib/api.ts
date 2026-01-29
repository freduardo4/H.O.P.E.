const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000/api';

export async function fetchMarketplaceListings() {
    try {
        const res = await fetch(`${API_URL}/marketplace/listings`, { cache: 'no-store' });
        if (!res.ok) throw new Error('Failed to fetch listings');
        return await res.json();
    } catch (error) {
        console.error('Error fetching marketplace listings:', error);
        return [];
    }
}

export async function fetchWikiPosts(query: string = '') {
    try {
        const url = new URL(`${API_URL}/wiki-fix/posts`);
        if (query) url.searchParams.append('search', query);

        const res = await fetch(url.toString(), { cache: 'no-store' });
        if (!res.ok) throw new Error('Failed to fetch wiki posts');
        return await res.json();
    } catch (error) {
        console.error('Error fetching wiki posts:', error);
        return [];
    }
}
