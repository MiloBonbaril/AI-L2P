interface EvalBarProps {
  modelWinProbWhite: number | null; // 0-1, converted to White's perspective
  sfEvalPawns: number | null;
}

function sigmoidPawnsToWhiteProb(pawns: number): number {
  return 1 / (1 + Math.exp((-pawns * 100) / 400));
}

export function EvalBar({ modelWinProbWhite, sfEvalPawns }: EvalBarProps) {
  const sfProb = sfEvalPawns != null ? sigmoidPawnsToWhiteProb(sfEvalPawns) : null;
  return (
    <div className="space-y-2 rounded border border-zinc-200 p-3 dark:border-zinc-800">
      <div className="mb-1 text-xs text-zinc-500">Eval (White&apos;s perspective)</div>
      <Meter label="Model" value={modelWinProbWhite} />
      <Meter
        label="Stockfish"
        value={sfProb}
        suffix={sfEvalPawns != null ? `${sfEvalPawns > 0 ? "+" : ""}${sfEvalPawns.toFixed(1)}` : "…"}
      />
    </div>
  );
}

function Meter({ label, value, suffix }: { label: string; value: number | null; suffix?: string }) {
  const pct = value != null ? Math.round(value * 100) : 50;
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-16 shrink-0 text-zinc-500">{label}</span>
      <div className="h-3 flex-1 overflow-hidden rounded bg-zinc-800">
        <div className="h-3 bg-zinc-100" style={{ width: `${pct}%` }} />
      </div>
      <span className="w-12 shrink-0 text-right text-zinc-500">{suffix ?? `${pct}%`}</span>
    </div>
  );
}
