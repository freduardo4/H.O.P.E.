import React from 'react';
import Card, { CardContent } from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import { fetchWikiPosts } from '@/lib/api';

interface KnowledgeNode {
    id: string;
    name: string;
    type: string;
    description: string;
}

interface WikiPost {
    id: string;
    title: string;
    content: string;
    authorName: string;
    createdAt: string;
    upvotes: number;
}

const MOCK_POSTS: WikiPost[] = [
    { id: '1', title: 'P0300 Misfire on cold start', content: 'Check coil packs and spark plugs. Often caused by carbon build-up on intake valves.', authorName: 'TurboJoe', createdAt: '2026-01-15', upvotes: 42 },
    { id: '2', title: 'Unexpected Voltage Drop during Flashing', content: 'Always ensure a battery stabilizer is connected. Critical for VAG and BMW platforms.', authorName: 'H.O.P.E_Tech', createdAt: '2026-01-20', upvotes: 128 },
];

const MOCK_GRAPH: KnowledgeNode[] = [
    { id: 'n1', name: 'Spark Plug', type: 'PART', description: 'Ignition component. Wear can cause misfires.' },
    { id: 'n2', name: 'Coil Pack', type: 'PART', description: 'High voltage transformer. Common failure on TSI engines.' },
    { id: 'n3', name: 'Rough Idle', type: 'SYMPTOM', description: 'Uneven engine RPM at standstill.' },
];

export default async function WikiFixPage({ searchParams }: { searchParams: { q?: string } }) {
    const query = searchParams.q || "";
    const apiPosts = await fetchWikiPosts(query);
    const posts = (apiPosts && apiPosts.length > 0 ? apiPosts : MOCK_POSTS) as WikiPost[];
    const graph = MOCK_GRAPH; // Graph is still simulated

    return (
        <div className="p-8 md:p-16 max-w-7xl mx-auto flex flex-col lg:flex-row gap-12">
            <div className="flex-1">
                <header className="mb-12">
                    <h1 className="text-5xl font-black text-white mb-6 tracking-tight">
                        Wiki-<span className="gradient-text">Fix</span>
                    </h1>
                    <form className="flex gap-3 glass p-1.5 rounded-2xl border-white/10">
                        <input
                            name="q"
                            defaultValue={query}
                            placeholder="Search symptoms, DTCs (P0300), or hardware ID..."
                            className="flex-1 bg-transparent border-none px-6 py-4 text-white focus:outline-none placeholder:text-gray-500"
                        />
                        <Button className="rounded-xl px-10">SEARCH</Button>
                    </form>
                </header>

                <div className="space-y-6">
                    <h3 className="text-xs font-black uppercase tracking-[0.2em] text-gray-500 mb-4">Community Discussions</h3>
                    {posts.map(post => (
                        <Card key={post.id} className="bg-white/[0.02]">
                            <CardContent className="p-8">
                                <h2 className="text-2xl font-bold text-white mb-3 hover:text-primary transition-colors cursor-pointer">{post.title}</h2>
                                <p className="text-gray-400 mb-8 leading-relaxed">{post.content}</p>
                                <div className="flex justify-between items-center text-xs">
                                    <div className="flex items-center gap-2">
                                        <div className="w-6 h-6 rounded-full bg-gray-800 border border-white/10"></div>
                                        <span className="text-gray-500">By <strong className="text-gray-300">{post.authorName}</strong> • {post.createdAt}</span>
                                    </div>
                                    <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
                                        <span className="text-emerald-400 font-black">▲ {post.upvotes}</span>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            </div>

            <aside className="lg:w-96">
                <div className="sticky top-32">
                    <h3 className="text-xs font-black uppercase tracking-[0.2em] text-primary mb-8 border-b border-primary/20 pb-4">Knowledge Graph</h3>
                    <div className="space-y-4">
                        {graph.map(node => (
                            <Card key={node.id} className="bg-white/5 group transition-transform">
                                <CardContent className="p-5">
                                    <div className="flex justify-between items-start mb-2">
                                        <span className="text-sm font-black text-secondary tracking-tight">{node.name}</span>
                                        <span className="text-[10px] text-gray-600 font-mono bg-black/40 px-1.5 py-0.5 rounded">{node.type}</span>
                                    </div>
                                    <p className="text-xs text-gray-500 leading-relaxed font-medium">{node.description}</p>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                </div>
            </aside>
        </div>
    );
}
