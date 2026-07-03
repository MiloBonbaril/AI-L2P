import { b64ToBytes } from "@/lib/types";
import type { TraceLayer } from "@/lib/types";

interface BrainPanelProps {
  layers: TraceLayer[];
  currentLayer: number; // -1 = show everything at full opacity
}

export function BrainPanel({ layers, currentLayer }: BrainPanelProps) {
  return (
    <div className="flex flex-col gap-1">
      {layers.map((layer, i) => {
        const bytes = b64ToBytes(layer.brain);
        const revealed = currentLayer < 0 || i <= currentLayer;
        return (
          <div key={i} className="flex items-center gap-2">
            <span className="w-8 shrink-0 text-xs text-zinc-500">L{i + 1}</span>
            <div className="grid flex-1 gap-px" style={{ gridTemplateColumns: `repeat(${bytes.length}, 1fr)` }}>
              {Array.from(bytes).map((v, j) => (
                <div
                  key={j}
                  className="transition-opacity duration-200"
                  style={{
                    height: 14,
                    backgroundColor: "rgb(42, 120, 214)",
                    opacity: revealed ? v / 255 : 0,
                  }}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
