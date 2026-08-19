// For Vercel deployments and rewrites, we can just use relative paths
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "/api";

export async function fetchOverview() {
  const res = await fetch(`${API_BASE_URL}/overview`, { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error("Failed to fetch overview");
  return res.json();
}

export async function fetchClusters() {
  const res = await fetch(`${API_BASE_URL}/clusters`, { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error("Failed to fetch clusters");
  return res.json();
}

export async function fetchClusterDetail(id: number) {
  const res = await fetch(`${API_BASE_URL}/clusters/${id}`, { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error(`Failed to fetch cluster ${id}`);
  return res.json();
}

export async function fetchTitles(params: Record<string, any>) {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      query.append(key, String(value));
    }
  }
  const res = await fetch(`${API_BASE_URL}/titles?${query.toString()}`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch titles");
  return res.json();
}

export async function fetchFilters() {
  const res = await fetch(`${API_BASE_URL}/filters`, { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error("Failed to fetch filters");
  return res.json();
}

export async function fetchEvaluation() {
  const res = await fetch(`${API_BASE_URL}/evaluation`, { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error("Failed to fetch evaluation metrics");
  return res.json();
}

export async function fetchVisualization(mode: "2d" | "3d" = "2d") {
  const res = await fetch(`${API_BASE_URL}/visualization?mode=${mode}`, { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error("Failed to fetch visualization");
  return res.json();
}
