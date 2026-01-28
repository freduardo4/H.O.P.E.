// This is an App Router page (Server Component by default)

interface CmsContent {
  id: string;
  title: string;
  body: string;
  category: string;
  publishedAt: string;
}

async function getCmsContent(): Promise<CmsContent[]> {
  // In a real app, this would be:
  // const res = await fetch('http://localhost:3000/cms/content', { next: { revalidate: 60 } });
  // return res.json();

  // Simulating backend fetch for now
  return [
    {
      id: '1',
      title: 'Welcome to H.O.P.E. Central',
      body: 'Your journey with Advanced Agentic Coding starts here. Explore our Wiki-Fix and Calibration Marketplace.',
      category: 'announcement',
      publishedAt: new Date().toISOString(),
    },
    {
      id: '2',
      title: 'New Update: J2534 Stability Improvements',
      body: 'The latest desktop client update includes significant stability improvements for high-frequency data streaming.',
      category: 'news',
      publishedAt: new Date().toISOString(),
    },
  ];
}

export default async function Home() {
  const content = await getCmsContent();

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24 bg-[#1A1A1A] text-white">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <h1 className="text-4xl font-bold mb-8">H.O.P.E. Central</h1>
        <a href="/marketplace" className="text-[#00AAFF] hover:underline mb-8 lg:mb-0">Browse Marketplace &rarr;</a>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-5xl">
        {content.map((item) => (
          <div key={item.id} className="p-6 border border-gray-700 rounded-lg bg-[#252525] hover:bg-[#303030] transition-colors">
            <span className="text-xs uppercase tracking-widest text-[#DB4437] mb-2 block">{item.category}</span>
            <h2 className="text-xl font-semibold mb-4">{item.title}</h2>
            <p className="text-gray-400">{item.body}</p>
            <div className="mt-4 text-xs text-gray-600">{new Date(item.publishedAt).toLocaleDateString()}</div>
          </div>
        ))}
      </div>
    </main>
  );
}
