"use client";

import { useEffect, useState } from "react";
import { fetchTitles, fetchFilters } from "@/lib/api";
import { Search, Filter, ChevronLeft, ChevronRight } from "lucide-react";
import clsx from "clsx";

export default function ExplorerPage() {
  const [data, setData] = useState<any>({ results: [], total: 0, total_pages: 1, page: 1 });
  const [filters, setFilters] = useState<any>(null);
  
  // State
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [cluster, setCluster] = useState("");
  const [type, setType] = useState("");
  const [rating, setRating] = useState("");
  const [genre, setGenre] = useState("");
  const [loading, setLoading] = useState(true);

  // Load filters once
  useEffect(() => {
    fetchFilters().then(setFilters).catch(console.error);
  }, []);

  // Load data when filters change
  useEffect(() => {
    setLoading(true);
    const params = {
      page,
      limit: 20,
      search: search.trim() || undefined,
      cluster: cluster || undefined,
      type: type || undefined,
      rating: rating || undefined,
      genre: genre || undefined
    };
    
    // Debounce search slightly
    const timer = setTimeout(() => {
      fetchTitles(params)
        .then(setData)
        .catch(console.error)
        .finally(() => setLoading(false));
    }, 300);
    
    return () => clearTimeout(timer);
  }, [page, search, cluster, type, rating, genre]);

  return (
    <div className="p-8 max-w-7xl mx-auto h-full flex flex-col animate-in fade-in duration-500">
      <header className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight mb-2">Data Explorer</h1>
        <p className="text-muted">Search and filter the segmented Netflix catalog.</p>
      </header>

      {/* Filters Bar */}
      <div className="glass-panel p-4 rounded-xl mb-6 flex flex-wrap gap-4 items-center">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" />
          <input 
            type="text" 
            placeholder="Search titles or directors..." 
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1); }}
            className="w-full bg-surface border border-border rounded-lg pl-9 pr-4 py-2 text-sm focus:outline-none focus:border-accent text-foreground"
          />
        </div>

        <select 
          value={cluster} 
          onChange={(e) => { setCluster(e.target.value); setPage(1); }}
          className="bg-surface border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-accent text-foreground"
        >
          <option value="">All Clusters</option>
          {filters?.clusters?.map((c: any) => (
            <option key={c.id} value={c.id}>Cluster {c.id}</option>
          ))}
        </select>

        <select 
          value={type} 
          onChange={(e) => { setType(e.target.value); setPage(1); }}
          className="bg-surface border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-accent text-foreground"
        >
          <option value="">All Types</option>
          <option value="Movie">Movies</option>
          <option value="TV Show">TV Shows</option>
        </select>
        
        <select 
          value={genre} 
          onChange={(e) => { setGenre(e.target.value); setPage(1); }}
          className="bg-surface border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-accent text-foreground"
        >
          <option value="">All Genres</option>
          {filters?.top_genres?.map((g: string) => (
            <option key={g} value={g}>{g}</option>
          ))}
        </select>
      </div>

      {/* Table */}
      <div className="flex-1 glass-panel rounded-xl overflow-hidden flex flex-col border border-border">
        <div className="overflow-x-auto flex-1">
          <table className="w-full text-left text-sm whitespace-nowrap">
            <thead className="bg-surface text-muted text-xs uppercase sticky top-0 z-10 border-b border-border">
              <tr>
                <th className="px-6 py-4 font-medium">Title</th>
                <th className="px-6 py-4 font-medium">Cluster</th>
                <th className="px-6 py-4 font-medium">Type</th>
                <th className="px-6 py-4 font-medium">Release Year</th>
                <th className="px-6 py-4 font-medium">Rating</th>
                <th className="px-6 py-4 font-medium">Genres</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {loading ? (
                <tr>
                  <td colSpan={6} className="px-6 py-12 text-center text-muted">
                    Loading results...
                  </td>
                </tr>
              ) : data.results.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-6 py-12 text-center text-muted">
                    No titles match your filters.
                  </td>
                </tr>
              ) : (
                data.results.map((row: any) => (
                  <tr key={row.show_id} className="hover:bg-surfaceHover/50 transition-colors">
                    <td className="px-6 py-3 font-medium text-foreground max-w-[250px] truncate" title={row.title}>
                      {row.title}
                    </td>
                    <td className="px-6 py-3">
                      <span className={`inline-flex items-center justify-center px-2 py-0.5 text-xs font-bold rounded bg-cluster-${row.cluster_id}/20 text-cluster-${row.cluster_id}`}>
                        C{row.cluster_id}
                      </span>
                    </td>
                    <td className="px-6 py-3 text-muted">{row.type}</td>
                    <td className="px-6 py-3 text-muted">{row.release_year}</td>
                    <td className="px-6 py-3 text-muted">{row.rating}</td>
                    <td className="px-6 py-3 text-muted max-w-[300px] truncate" title={row.listed_in}>
                      {row.listed_in}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
        
        {/* Pagination */}
        <div className="bg-surface p-4 border-t border-border flex items-center justify-between text-sm shrink-0">
          <div className="text-muted">
            Showing <span className="font-medium text-foreground">{data.total > 0 ? (page - 1) * 20 + 1 : 0}</span> to <span className="font-medium text-foreground">{Math.min(page * 20, data.total)}</span> of <span className="font-medium text-foreground">{data.total}</span> results
          </div>
          <div className="flex space-x-2">
            <button 
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              className="p-2 rounded border border-border disabled:opacity-50 hover:bg-surfaceHover disabled:hover:bg-transparent"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button 
              onClick={() => setPage(p => Math.min(data.total_pages, p + 1))}
              disabled={page === data.total_pages || data.total_pages === 0}
              className="p-2 rounded border border-border disabled:opacity-50 hover:bg-surfaceHover disabled:hover:bg-transparent"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
