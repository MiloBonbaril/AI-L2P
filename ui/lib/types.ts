export interface LogitTopK {
  move: string;
  prob: number;
}

export interface TraceLayer {
  attn: string; // base64 uint8[heads*64]
  attn_heads: number;
  brain: string; // base64 uint8[64]
  logits_topk: LogitTopK[];
}

export interface TraceMessage {
  fen: string;
  chosen_move: string;
  legal_moves: string[];
  per_layer: TraceLayer[];
  value_bucket_dist: string; // base64 uint8[64], normalized for display only
  value_winprob: number; // real expected win probability, 0-1
}

export interface InitMessage {
  type: "init";
  fen: string;
  model_color: "white" | "black";
  opponent: string;
}

export interface MoveMessage {
  type: "move";
  color: "white" | "black";
  uci: string;
  fen: string;
  trace: TraceMessage | null;
  sf_eval: number | null;
}

export interface GameOverMessage {
  type: "game_over";
  result: string;
}

export type ServerMessage = InitMessage | MoveMessage | GameOverMessage;

export function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}
