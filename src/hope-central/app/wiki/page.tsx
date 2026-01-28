import React from 'react';

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

async function search(query: string = "") {
    // Simulating GraphQL/API fetch
    return {
        posts: [
            { id: '1', title: 'P0300 Misfire on cold start', content: 'Check coil packs and spark plugs. Often caused by carbon build-up on intake valves.', authorName: 'TurboJoe', createdAt: '2026-01-15', upvotes: 42 },
        ] as WikiPost[],
        graph: [
            { id: 'n1', name: 'Spark Plug', type: 'PART', description: 'Ignition component. Wear can cause misfires.' },
            { id: 'n2', name: 'Coil Pack', type: 'PART', description: 'High voltage transformer. Common failure on TSI engines.' },
            { id: 'n3', name: 'Rough Idle', type: 'SYMPTOM', description: 'Uneven engine RPM at standstill.' },
        ] as KnowledgeNode[]
    };
}

export default async function WikiFixPage({ searchParams }: { searchParams: { q?: string } }) {
    const query = searchParams.q || "";
    const { posts, graph } = await search(query);

    return (
        <div className="min-h-screen bg-[#0A0A0A] text-white p-8 md:p-24 flex">
            <div className="flex-1 mr-12">
                <header className="mb-12">
                    <h1 className="text-4xl font-bold mb-4">Wiki-Fix Community</h1>
                    <form className="flex gap-4">
                        <input
                            name="q"
                            defaultValue={query}
                            placeholder="Search by symptom or DTC (e.g., 'P0300')..."
                            className="flex-1 bg-[#1A1A1A] border border-gray-700 rounded-lg px-4 py-2 focus:border-[#00AAFF] outline-none"
                        />
                        <button className="bg-[#00AAFF] px-8 py-2 rounded-lg font-bold">SEARCH</button>
                    </form>
                </header>

                <section className="space-y-6">
                    {posts.map(post => (
                        <div key={post.id} className="bg-[#151515] border border-gray-800 rounded-xl p-6">
                            <h2 className="text-xl font-bold text-[#00AAFF] mb-2">{post.title}</h2>
                            <p className="text-gray-400 mb-6">{post.content}</p>
                            <div className="flex justify-between items-center text-xs text-gray-500">
                                <span>By <strong>{post.authorName}</strong> • {post.createdAt}</span>
                                <span className="text-[#00FF00] font-bold">▲ {post.upvotes}</span>
                            </div>
                        </div>
                    ))}
                </section>
            </div>

            <aside className="w-80 border-l border-gray-800 pl-8">
                <h3 className="text-sm font-bold text-[#00AAFF] mb-6 uppercase tracking-wider">Knowledge Graph</h3>
                <div className="space-y-4">
                    {graph.map(node => (
                        <div key={node.id} className="bg-[#1A1A1A] border border-gray-800 rounded-lg p-4">
                            <div className="flex justify-between items-start mb-2">
                                <span className="text-sm font-bold text-[#FFAA00]">{node.name}</span>
                                <span className="text-[10px] text-gray-600 font-mono">{node.type}</span>
                            </div>
                            <p className="text-xs text-gray-500 leading-relaxed">{node.description}</p>
                        </div>
                    ))}
                </div>
            </aside>
        </div>
    );
}
