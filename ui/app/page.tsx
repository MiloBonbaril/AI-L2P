"use client";

import { useState } from "react";
import { useExhibition } from "@/lib/useExhibition";
import { Board } from "@/components/Board";
import { BrainPanel } from "@/components/BrainPanel";
import { LogitLens } from "@/components/LogitLens";
import { EvalBar } from "@/components/EvalBar";
import { b64ToBytes } from "@/lib/types";

const WS_URL = process.env.NEXT_PUBLIC_EXHIBITION_WS ?? "ws://127.0.0.1:8420/ws/exhibition";

export default function Home() {
  const [selectedHead, setSelectedHead] = useState<number | "avg">("avg");
  const { status, fen, lastMove, trace, currentLayer, isAnimating, sfEval, gameOver, moveColor, requestNext } =
    useExhibition(WS_URL, { opponent: "stockfish", modelColor: "white", elo: 1500 });

  const activeLayer = trace && currentLayer >= 0 ? trace.per_layer[currentLayer] : null;
  const attnBytes = activeLayer ? b64ToBytes(activeLayer.attn) : null;
  const heads = activeLayer?.attn_heads ?? 8;

  const modelWinProbWhite = trace ? (moveColor === "black" ? 1 - trace.value_winprob : trace.value_winprob) : null;

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-6 p-8">
      <header className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">Glass Knight &mdash; Exhibition</h1>
        <span className="text-xs text-zinc-500">{status}</span>
      </header>

      <div className="flex flex-wrap gap-8">
        <div className="flex flex-col gap-3">
          <Board fen={fen} lastMove={lastMove} attn={attnBytes} heads={heads} selectedHead={selectedHead} />

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-zinc-500">Head:</span>
            <HeadButton active={selectedHead === "avg"} onClick={() => setSelectedHead("avg")}>
              avg
            </HeadButton>
            {Array.from({ length: heads }).map((_, h) => (
              <HeadButton key={h} active={selectedHead === h} onClick={() => setSelectedHead(h)}>
                {h}
              </HeadButton>
            ))}
          </div>

          <button
            onClick={requestNext}
            disabled={isAnimating || status !== "open" || !!gameOver}
            className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-opacity disabled:opacity-40"
          >
            {isAnimating ? "thinking…" : "Next move"}
          </button>
          {gameOver && <div className="text-sm text-zinc-500">Game over: {gameOver}</div>}
        </div>

        <div className="flex w-80 flex-col gap-4">
          <EvalBar modelWinProbWhite={modelWinProbWhite} sfEvalPawns={sfEval} />
          <LogitLens layer={activeLayer} layerIndex={Math.max(currentLayer, 0)} totalLayers={trace?.per_layer.length ?? 8} />
        </div>
      </div>

      {trace && (
        <div>
          <h2 className="mb-2 text-sm text-zinc-500">Brain activity</h2>
          <BrainPanel layers={trace.per_layer} currentLayer={currentLayer} />
        </div>
      )}
    </div>
  );
}

function HeadButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`rounded px-2 py-0.5 ${active ? "bg-blue-600 text-white" : "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400"}`}
    >
      {children}
    </button>
  );
}
