import type { TraceLayer } from "@/lib/types";

interface LogitLensProps {
  layer: TraceLayer | null;
  layerIndex: number;
  totalLayers: number;
}

export function LogitLens({ layer, layerIndex, totalLayers }: LogitLensProps) {
  return (
    <div className="rounded border border-zinc-200 p-3 dark:border-zinc-800">
      <div className="mb-2 text-xs text-zinc-500">
        layer {layerIndex + 1} / {totalLayers} &middot; candidate moves
      </div>
      <ol className="space-y-1.5">
        {(layer?.logits_topk ?? []).map((c, i) => (
          <li key={c.move} className="flex items-center gap-2 text-sm">
            <span className="w-4 text-zinc-400">{i + 1}</span>
            <span className="w-14 font-mono">{c.move}</span>
            <div className="h-2 flex-1 overflow-hidden rounded bg-zinc-100 dark:bg-zinc-800">
              <div className="h-2 rounded bg-blue-600" style={{ width: `${Math.round(c.prob * 100)}%` }} />
            </div>
            <span className="w-10 text-right text-zinc-500">{Math.round(c.prob * 100)}%</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
