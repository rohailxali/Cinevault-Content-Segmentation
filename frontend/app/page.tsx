import { fetchOverview } from "@/lib/api";
import { Film, MonitorPlay, Layers, Target, PlayCircle, BarChart3 } from "lucide-react";

export const revalidate = 3600;

function StatCard({ title, value, icon: Icon, subtitle }: any) {
  return (
    <div className="glass-panel p-6 rounded-xl flex items-start justify-between">
      <div>
        <div className="text-sm font-medium text-muted mb-1">{title}</div>
        <div className="text-3xl font-bold text-foreground mb-1">{value}</div>
        {subtitle && <div className="text-xs text-muted">{subtitle}</div>}
      </div>
      <div className="p-3 bg-surface rounded-lg text-accent">
        <Icon className="w-6 h-6" />
      </div>
    </div>
  );
}

export default async function OverviewPage() {
  let overview;
  try {
    overview = await fetchOverview();
  } catch (err) {
    return (
      <div className="p-8">
        <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-lg">
          Failed to load model artifacts. Ensure the ML pipeline and API backend are running.
        </div>
      </div>
    );
  }

  const m = overview.clustering_metrics;

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <header>
        <h1 className="text-3xl font-bold tracking-tight mb-2">Platform Overview</h1>
        <p className="text-muted text-lg">
          Discover meaningful content segments from the Netflix catalog using unsupervised machine learning.
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard 
          title="Total Titles" 
          value={overview.total_titles.toLocaleString()} 
          icon={Film}
          subtitle={`Years: ${overview.dataset_year_range[0]} - ${overview.dataset_year_range[1]}`}
        />
        <StatCard 
          title="Movies" 
          value={overview.n_movies.toLocaleString()} 
          icon={PlayCircle}
          subtitle={`${Math.round(overview.n_movies / overview.total_titles * 100)}% of dataset`}
        />
        <StatCard 
          title="TV Shows" 
          value={overview.n_tv_shows.toLocaleString()} 
          icon={MonitorPlay}
          subtitle={`${Math.round(overview.n_tv_shows / overview.total_titles * 100)}% of dataset`}
        />
        <StatCard 
          title="Optimal Clusters" 
          value={overview.n_clusters} 
          icon={Layers}
          subtitle={`K-Means algorithm`}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-panel p-6 rounded-xl">
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <Target className="w-5 h-5 mr-2 text-accent" />
            Cluster Quality (K={overview.selected_k})
          </h2>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-muted">Silhouette Score (Cohesion & Separation)</span>
                <span className="font-medium text-green-400">{m.silhouette_score.toFixed(4)}</span>
              </div>
              <div className="w-full bg-surface rounded-full h-2">
                <div className="bg-green-500 h-2 rounded-full" style={{ width: `${Math.max(0, m.silhouette_score * 100)}%` }}></div>
              </div>
              <p className="text-xs text-muted mt-1">Values closer to 1 indicate better separation.</p>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-muted">Davies-Bouldin Index</span>
                <span className="font-medium text-yellow-400">{m.davies_bouldin_score.toFixed(4)}</span>
              </div>
              <p className="text-xs text-muted mt-1">Lower values indicate better clustering.</p>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-muted">Calinski-Harabasz Score</span>
                <span className="font-medium text-blue-400">{Math.round(m.calinski_harabasz_score).toLocaleString()}</span>
              </div>
              <p className="text-xs text-muted mt-1">Higher values indicate better defined clusters.</p>
            </div>
          </div>
        </div>

        <div className="glass-panel p-6 rounded-xl">
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <BarChart3 className="w-5 h-5 mr-2 text-accent" />
            Segment Distribution
          </h2>
          <div className="space-y-4">
            {overview.cluster_sizes.map(([id, count]: [number, number], i: number) => {
              const pct = (count / overview.total_titles) * 100;
              return (
                <div key={id}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="font-medium text-foreground">Cluster {id} <span className="text-muted font-normal ml-1">— {overview.cluster_labels[id]}</span></span>
                    <span className="text-muted">{count.toLocaleString()} ({pct.toFixed(1)}%)</span>
                  </div>
                  <div className="w-full bg-surface rounded-full h-2">
                    <div className={`h-2 rounded-full bg-cluster-${id}`} style={{ width: `${pct}%` }}></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
      
      <div className="glass-panel p-6 rounded-xl">
        <h2 className="text-xl font-bold mb-4">Dataset Health</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-surface rounded-lg">
            <div className="text-sm text-muted">Director Information Missing</div>
            <div className="text-2xl font-bold mt-1 text-yellow-500">{overview.data_quality.director_not_given_pct}%</div>
            <div className="text-xs text-muted mt-1">Imputed/excluded from core features</div>
          </div>
          <div className="p-4 bg-surface rounded-lg">
            <div className="text-sm text-muted">Country Information Missing</div>
            <div className="text-2xl font-bold mt-1 text-yellow-500">{overview.data_quality.country_not_given_pct}%</div>
            <div className="text-xs text-muted mt-1">Imputed/excluded from core features</div>
          </div>
        </div>
      </div>
    </div>
  );
}
