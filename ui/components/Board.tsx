"use client";

import { useEffect, useRef } from "react";
import { Chessground } from "chessground";
import type { Api } from "chessground/api";
import type { Key } from "chessground/types";

interface BoardProps {
  fen: string;
  lastMove: [string, string] | null;
  attn: Uint8Array | null; // heads*64, MOVE-token attention to each square
  heads: number;
  selectedHead: number | "avg";
}

const SIZE = 480;

export function Board({ fen, lastMove, attn, heads, selectedHead }: BoardProps) {
  const el = useRef<HTMLDivElement>(null);
  const api = useRef<Api | null>(null);

  useEffect(() => {
    if (!el.current) return;
    api.current = Chessground(el.current, {
      fen,
      viewOnly: true,
      coordinates: true,
      animation: { enabled: true, duration: 200 },
    });
    return () => api.current?.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    api.current?.set({ fen, lastMove: (lastMove as Key[] | null) ?? undefined });
  }, [fen, lastMove]);

  const cellValue = (squareIdx: number): number => {
    if (!attn) return 0;
    if (selectedHead === "avg") {
      let sum = 0;
      for (let h = 0; h < heads; h++) sum += attn[h * 64 + squareIdx];
      return sum / heads;
    }
    return attn[selectedHead * 64 + squareIdx];
  };

  return (
    <div className="relative" style={{ width: SIZE, height: SIZE }}>
      <div ref={el} style={{ width: SIZE, height: SIZE }} />
      <div className="pointer-events-none absolute inset-0 grid grid-cols-8 grid-rows-8">
        {Array.from({ length: 64 }).map((_, idx) => {
          const file = idx % 8;
          const rank = Math.floor(idx / 8);
          const row = 7 - rank; // rank8 at the top for white orientation
          const value = cellValue(idx);
          const alpha = (value / 255) * 0.7;
          return (
            <div
              key={idx}
              style={{
                gridRow: row + 1,
                gridColumn: file + 1,
                backgroundColor: `rgba(42, 120, 214, ${alpha})`,
              }}
            />
          );
        })}
      </div>
    </div>
  );
}
