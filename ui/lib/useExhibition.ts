"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { GameOverMessage, MoveMessage, ServerMessage, TraceMessage } from "./types";

const LAYER_MS = 350; // ~2.8s across 8 layers -- the "2-4 second" pacing trick (concept doc 7.1)
const HOLD_MS = 500; // pause on the final layer before committing the move

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export interface ExhibitionOptions {
  opponent?: "stockfish" | "self";
  modelColor?: "white" | "black";
  elo?: number;
}

export function useExhibition(wsUrl: string, opts: ExhibitionOptions = {}) {
  const { opponent = "stockfish", modelColor = "white", elo = 1500 } = opts;

  const [status, setStatus] = useState<"connecting" | "open" | "closed">("connecting");
  const [fen, setFen] = useState(START_FEN);
  const [lastMove, setLastMove] = useState<[string, string] | null>(null);
  const [trace, setTrace] = useState<TraceMessage | null>(null);
  const [currentLayer, setCurrentLayer] = useState(-1);
  const [isAnimating, setIsAnimating] = useState(false);
  const [sfEval, setSfEval] = useState<number | null>(null);
  const [gameOver, setGameOver] = useState<string | null>(null);
  const [moveColor, setMoveColor] = useState<"white" | "black" | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const commit = useCallback((msg: MoveMessage) => {
    setFen(msg.fen);
    setLastMove([msg.uci.slice(0, 2), msg.uci.slice(2, 4)]);
    setSfEval(msg.sf_eval);
    setMoveColor(msg.color);
    setIsAnimating(false);
  }, []);

  const animateThenCommit = useCallback(
    (msg: MoveMessage) => {
      const layerCount = msg.trace!.per_layer.length;
      setTrace(msg.trace);
      setIsAnimating(true);
      setCurrentLayer(0);

      let layer = 0;
      const step = () => {
        layer += 1;
        if (layer < layerCount) {
          setCurrentLayer(layer);
          timerRef.current = setTimeout(step, LAYER_MS);
        } else {
          timerRef.current = setTimeout(() => commit(msg), HOLD_MS);
        }
      };
      timerRef.current = setTimeout(step, LAYER_MS);
    },
    [commit],
  );

  useEffect(() => {
    const url = `${wsUrl}?opponent=${opponent}&model_color=${modelColor}&elo=${elo}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setStatus("open");
    ws.onclose = () => setStatus("closed");
    ws.onmessage = (ev) => {
      const msg: ServerMessage = JSON.parse(ev.data);
      if (msg.type === "init") {
        setFen(msg.fen);
      } else if (msg.type === "move") {
        if (msg.trace) {
          animateThenCommit(msg);
        } else {
          commit(msg);
        }
      } else if (msg.type === "game_over") {
        setGameOver((msg as GameOverMessage).result);
      }
    };

    return () => {
      ws.close();
      if (timerRef.current) clearTimeout(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wsUrl, opponent, modelColor, elo]);

  const requestNext = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN && !isAnimating) {
      wsRef.current.send(JSON.stringify({ type: "next" }));
    }
  }, [isAnimating]);

  return { status, fen, lastMove, trace, currentLayer, isAnimating, sfEval, gameOver, moveColor, requestNext };
}
