"use client";

import { useEffect, useState } from "react";
import { fetchEvaluation } from "@/lib/api";
import dynamic from "next/dynamic";
import { CheckCircle2, TrendingUp, Key, Binary } from "lucide-react";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false, loading: () => <div className="w-full h-full flex items-center justify-center bg-surface animate-pulse rounded-lg border border-border">Loading Metrics...</div> });

export default function EvaluationPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchEvaluation().then(setData).catch(console.error).finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center h-full">
        <div className="animate-pulse flex flex-col items-center">
          <div className="w-12 h-12 border-4 border-accent border-t-transparent rounded-full animate-spin mb-4"></div>
          <p className="text-muted">Loading model evaluation metrics...</p>
        </div>
      </div>
    );
  }

  if (!data) return <div className="p-8 text-red-500">Failed to load evaluation metrics.</div>;

  const kRange = data.k_range;
  
  const lineChartLayout = {
    autosize: true,
    margin: { l: 40, r: 20, t: 20, b: 40 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    xaxis: { 
      gridcolor: '#2a2a35', 
      zerolinecolor: '#3a3a45',
      tickfont: { color: '#9b9bab' },
      dtick: 1
    },
    yaxis: { 
      gridcolor: '#2a2a35', 
      zerolinecolor: '#3a3a45',
      tickfont: { color: '#9b9bab' }
    },
    hovermode: 'x unified' as const,
  };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
      <header>
        <h1 className="text-3xl font-bold tracking-tight mb-2">Model Evaluation</h1>
        <p className="text-muted">Metrics used to determine the optimal number of clusters (K).</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="glass-panel p-6 rounded-xl flex flex-col justify-center">
          <div className="text-sm font-bold uppercase tracking-wider text-muted mb-4 flex items-center">
            <CheckCircle2 className="w-4 h-4 mr-2" /> Selected Architecture
          </div>
          <div className="text-6xl font-bold text-accent mb-2">K = {data.selected_k}</div>
          <p className="text-sm text-foreground mb-4">
            Optimal segmentation based on composite scoring and analytical interpretability. 
            Balancing mathematical cohesion (Silhouette) with semantic value.
          </p>
          <div className="pt-4 border-t border-border mt-auto">
            <div className="text-xs text-muted mb-1">Algorithm</div>
            <div className="font-mono text-sm">MiniBatchKMeans (init: k-means++)</div>
          </div>
        </div>

        <div className="glass-panel p-6 rounded-xl col-span-1 lg:col-span-2">
          <h3 className="text-sm font-bold uppercase tracking-wider text-muted mb-4 flex items-center">
            <Binary className="w-4 h-4 mr-2" /> Feature Space
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-3xl font-bold mb-1">{data.feature_names.length}</div>
              <div className="text-sm text-muted">Engineered Features</div>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-border">
            <p className="text-xs text-muted mb-2">Feature Breakdown:</p>
            <div className="flex flex-wrap gap-2">
              <span className="px-2 py-1 bg-surface rounded text-xs">Release Year (Scaled)</span>
              <span className="px-2 py-1 bg-surface rounded text-xs">Movie Duration (Min)</span>
              <span className="px-2 py-1 bg-surface rounded text-xs">TV Seasons</span>
              <span className="px-2 py-1 bg-surface rounded text-xs text-accent">is_movie (Binary)</span>
              <span className="px-2 py-1 bg-surface rounded text-xs">Ratings (One-Hot)</span>
              <span className="px-2 py-1 bg-surface rounded text-xs">Genres (Multi-Hot)</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-panel p-6 rounded-xl h-[400px] flex flex-col">
          <h3 className="text-lg font-bold mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-accent" /> Silhouette Score vs K
          </h3>
          <div className="flex-1 min-h-0">
            <Plot
              data={[{
                x: kRange,
                y: data.k_evaluations.map((e: any) => e.silhouette),
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#10b981', width: 3 },
                marker: { size: 8 }
              }]}
              layout={lineChartLayout}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
              config={{ displayModeBar: false }}
            />
          </div>
          <p className="text-xs text-muted mt-2 text-center">Higher is better. Measures how similar an object is to its own cluster compared to others.</p>
        </div>

        <div className="glass-panel p-6 rounded-xl h-[400px] flex flex-col">
          <h3 className="text-lg font-bold mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-accent" /> Davies-Bouldin vs K
          </h3>
          <div className="flex-1 min-h-0">
            <Plot
              data={[{
                x: kRange,
                y: data.k_evaluations.map((e: any) => e.davies_bouldin),
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#f59e0b', width: 3 },
                marker: { size: 8 }
              }]}
              layout={lineChartLayout}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
              config={{ displayModeBar: false }}
            />
          </div>
          <p className="text-xs text-muted mt-2 text-center">Lower is better. Measures the ratio of within-cluster distances to between-cluster distances.</p>
        </div>
        
        <div className="glass-panel p-6 rounded-xl h-[400px] flex flex-col md:col-span-2">
          <h3 className="text-lg font-bold mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-accent" /> Inertia (Elbow Method)
          </h3>
          <div className="flex-1 min-h-0">
            <Plot
              data={[{
                x: kRange,
                y: data.k_evaluations.map((e: any) => e.inertia),
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#3b82f6', width: 3 },
                marker: { size: 8 }
              }]}
              layout={lineChartLayout}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
              config={{ displayModeBar: false }}
            />
          </div>
          <p className="text-xs text-muted mt-2 text-center">Lower is better. Sum of squared distances of samples to their closest cluster center. Look for the "elbow" point.</p>
        </div>
      </div>
    </div>
  );
}
