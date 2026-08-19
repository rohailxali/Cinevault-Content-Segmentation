"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";

// Plotly needs to be dynamically imported without SSR
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false, loading: () => <div className="w-full h-full flex items-center justify-center bg-surface animate-pulse rounded-lg border border-border">Loading Visualization...</div> });

export default function PCAChart({ points, selectedCluster, onPointClick }: { points: any[], selectedCluster: number | null, onPointClick?: (point: any) => void }) {
  
  const data = useMemo(() => {
    // Group by cluster
    const clusters: Record<number, any[]> = {};
    points.forEach(p => {
      if (!clusters[p.cluster_id]) clusters[p.cluster_id] = [];
      clusters[p.cluster_id].push(p);
    });

    const traces = Object.keys(clusters).map(clusterStr => {
      const clusterId = parseInt(clusterStr);
      const clusterPoints = clusters[clusterId];
      
      const isSelected = selectedCluster === null || selectedCluster === clusterId;
      const opacity = isSelected ? 0.8 : 0.1;
      
      const colors = [
        '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', 
        '#ec4899', '#06b6d4', '#f43f5e', '#84cc16'
      ];
      const color = colors[clusterId % colors.length];

      return {
        x: clusterPoints.map(p => p.pc1),
        y: clusterPoints.map(p => p.pc2),
        text: clusterPoints.map(p => `${p.title}<br>${p.type} | ${p.release_year}<br>${p.listed_in}`),
        customdata: clusterPoints.map(p => p.show_id),
        mode: 'markers',
        type: 'scattergl',
        name: `Cluster ${clusterId}`,
        marker: {
          color: color,
          size: isSelected ? 6 : 4,
          opacity: opacity,
          line: {
            color: 'rgba(255, 255, 255, 0.2)',
            width: isSelected ? 1 : 0
          }
        },
        hoverinfo: 'text'
      };
    });

    return traces;
  }, [points, selectedCluster]);

  return (
    <div className="w-full h-[600px] rounded-lg overflow-hidden border border-border bg-surface">
      <Plot
        data={data as any}
        layout={{
          autosize: true,
          margin: { l: 40, r: 20, t: 20, b: 40 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          xaxis: { 
            title: 'PC1',
            gridcolor: '#2a2a35', 
            zerolinecolor: '#3a3a45',
            tickfont: { color: '#9b9bab' },
            titlefont: { color: '#9b9bab' }
          },
          yaxis: { 
            title: 'PC2',
            gridcolor: '#2a2a35', 
            zerolinecolor: '#3a3a45',
            tickfont: { color: '#9b9bab' },
            titlefont: { color: '#9b9bab' }
          },
          legend: {
            font: { color: '#f5f5f7' },
            bgcolor: 'rgba(18, 18, 26, 0.8)',
            bordercolor: '#2a2a35',
            borderwidth: 1
          },
          hovermode: 'closest',
          dragmode: 'pan',
        }}
        useResizeHandler={true}
        style={{ width: '100%', height: '100%' }}
        config={{ displayModeBar: true, scrollZoom: true, displaylogo: false }}
        onClick={(data) => {
          if (onPointClick && data.points && data.points.length > 0) {
             const pt = data.points[0];
             onPointClick({ show_id: pt.customdata });
          }
        }}
      />
    </div>
  );
}
