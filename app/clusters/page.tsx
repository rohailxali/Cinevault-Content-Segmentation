"use client";

import { useEffect, useState } from "react";
import { fetchClusters, fetchVisualization } from "@/lib/api";
import PCAChart from "@/components/clusters/PCAChart";
import clsx from "clsx";
import { Users, Film, Tv, BarChart, ChevronRight } from "lucide-react";

export default function ClustersPage() {
  const [clusters, setClusters] = useState<any[]>([]);
  const [visualization, setVisualization] = useState<any>(null);
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([fetchClusters(), fetchVisualization()]).then(([cls, vis]) => {
      setClusters(cls);
      setVisualization(vis);
      setLoading(false);
    }).catch(console.error);
  }, []);

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center h-full">
        <div className="animate-pulse flex flex-col items-center">
          <div className="w-12 h-12 border-4 border-accent border-t-transparent rounded-full animate-spin mb-4"></div>
          <p className="text-muted">Loading cluster data & visualization...</p>
        </div>
      </div>
    );
  }

  const selectedProfile = selectedCluster !== null ? clusters.find(c => c.cluster_id === selectedCluster) : null;

  return (
    <div className="p-8 max-w-[1600px] mx-auto h-full flex flex-col animate-in fade-in duration-500">
      <header className="mb-6 flex justify-between items-end shrink-0">
        <div>
          <h1 className="text-3xl font-bold tracking-tight mb-2">Cluster Analysis</h1>
          <p className="text-muted">Explore content segments and their characteristic features in PCA space.</p>
        </div>
        <div className="text-sm bg-surface px-4 py-2 rounded-full border border-border">
          <span className="text-muted">Explained Variance (2D):</span> <span className="font-bold text-foreground">{(visualization?.total_explained_variance * 100).toFixed(1)}%</span>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[600px]">
        {/* Left: Cluster List */}
        <div className="col-span-1 lg:col-span-3 flex flex-col space-y-4 overflow-y-auto pr-2 scrollbar-hide pb-10">
          <button
            onClick={() => setSelectedCluster(null)}
            className={clsx(
              "text-left p-4 rounded-xl border transition-all duration-300",
              selectedCluster === null
                ? "bg-surface border-accent shadow-[0_0_15px_rgba(220,38,38,0.15)]"
                : "glass-panel border-border hover:border-muted"
            )}
          >
            <h3 className="font-bold">All Clusters</h3>
            <p className="text-xs text-muted mt-1">View entire dataset</p>
          </button>

          {clusters.map((cluster) => (
            <button
              key={cluster.cluster_id}
              onClick={() => setSelectedCluster(cluster.cluster_id)}
              className={clsx(
                "text-left p-4 rounded-xl border transition-all duration-300 relative overflow-hidden group",
                selectedCluster === cluster.cluster_id
                  ? `bg-surface border-cluster-${cluster.cluster_id}`
                  : "glass-panel border-border hover:border-muted"
              )}
            >
              {selectedCluster === cluster.cluster_id && (
                <div className={`absolute top-0 left-0 w-1 h-full bg-cluster-${cluster.cluster_id}`}></div>
              )}
              
              <div className="flex justify-between items-start mb-2">
                <span className={`text-xs font-bold px-2 py-0.5 rounded-full bg-cluster-${cluster.cluster_id}/20 text-cluster-${cluster.cluster_id}`}>
                  C{cluster.cluster_id}
                </span>
                <span className="text-xs text-muted">{cluster.pct_of_dataset.toFixed(1)}%</span>
              </div>
              
              <h3 className="font-bold text-sm mb-1 leading-snug">{cluster.label}</h3>
              <div className="text-xs text-muted flex items-center mt-3">
                <Users className="w-3 h-3 mr-1" /> {cluster.n_titles.toLocaleString()} titles
              </div>
            </button>
          ))}
        </div>

        {/* Center: PCA Visualization */}
        <div className="col-span-1 lg:col-span-6 flex flex-col">
          {visualization && (
            <PCAChart 
              points={visualization.points} 
              selectedCluster={selectedCluster} 
            />
          )}
          {visualization?.sampled && (
            <p className="text-xs text-muted mt-2 text-center">
              Displaying a representative sample of 5,000 titles for performance.
            </p>
          )}
        </div>

        {/* Right: Selected Cluster Profile */}
        <div className="col-span-1 lg:col-span-3 flex flex-col overflow-y-auto pr-2 scrollbar-hide pb-10">
          {selectedProfile ? (
            <div className="glass-panel border-border rounded-xl p-5 h-full animate-in slide-in-from-right-4 duration-300">
              <div className="mb-6 pb-4 border-b border-border">
                <span className={`inline-block text-xs font-bold px-2 py-1 rounded bg-cluster-${selectedProfile.cluster_id}/20 text-cluster-${selectedProfile.cluster_id} mb-3`}>
                  Cluster {selectedProfile.cluster_id} Profile
                </span>
                <h2 className="text-xl font-bold mb-2">{selectedProfile.label}</h2>
                <div className="flex space-x-4 text-sm text-muted">
                  <span className="flex items-center"><Film className="w-4 h-4 mr-1"/> {selectedProfile.n_movies}</span>
                  <span className="flex items-center"><Tv className="w-4 h-4 mr-1"/> {selectedProfile.n_tv_shows}</span>
                </div>
              </div>

              <div className="space-y-6">
                <div>
                  <h4 className="text-xs uppercase font-bold text-muted mb-3 flex items-center">
                    <BarChart className="w-3 h-3 mr-2" /> Top Genres (Lift)
                  </h4>
                  <div className="space-y-3">
                    {selectedProfile.top_genres.map((g: any, i: number) => (
                      <div key={i}>
                        <div className="flex justify-between text-sm mb-1">
                          <span>{g.value}</span>
                          <span className={g.lift > 2 ? "text-green-400" : "text-muted"}>{g.lift}x</span>
                        </div>
                        <div className="w-full bg-surface rounded-full h-1.5 flex">
                          <div className="bg-accent h-1.5 rounded-l-full" style={{ width: `${g.cluster_pct}%` }}></div>
                          <div className="bg-white/20 h-1.5 rounded-r-full" style={{ width: `${g.global_pct}%` }}></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="text-xs uppercase font-bold text-muted mb-3">Distinguishing Characteristics</h4>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-start">
                      <ChevronRight className="w-4 h-4 text-accent mr-1 shrink-0 mt-0.5" />
                      <span>Median Release: <strong className="text-foreground">{selectedProfile.release_year_median}</strong></span>
                    </li>
                    <li className="flex items-start">
                      <ChevronRight className="w-4 h-4 text-accent mr-1 shrink-0 mt-0.5" />
                      <span>Dominant Content: <strong className="text-foreground">{selectedProfile.dominant_type} ({selectedProfile.dominant_type_pct}%)</strong></span>
                    </li>
                    {selectedProfile.movie_duration_median_min && (
                       <li className="flex items-start">
                         <ChevronRight className="w-4 h-4 text-accent mr-1 shrink-0 mt-0.5" />
                         <span>Movie Duration: <strong className="text-foreground">~{selectedProfile.movie_duration_median_min} min</strong></span>
                       </li>
                    )}
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            <div className="glass-panel border-border rounded-xl p-8 h-full flex flex-col items-center justify-center text-center text-muted">
              <Network className="w-12 h-12 mb-4 opacity-20" />
              <p>Select a cluster on the left to view its detailed profile and characteristic statistics.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Temporary icon for empty state since we didn't import Network at the top level of this file
function Network(props: any) {
  return (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="16" y="16" width="6" height="6" rx="1"></rect>
      <rect x="2" y="16" width="6" height="6" rx="1"></rect>
      <rect x="9" y="2" width="6" height="6" rx="1"></rect>
      <path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3"></path>
      <path d="M12 12V8"></path>
    </svg>
  );
}
