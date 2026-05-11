export type RouteDecision = "SIMPLE" | "COMPLEX" | "SENSITIVE";

export interface ChunkRef {
  source: string;
  page: number;
  text: string;
  score?: number;
}

export interface RouteInfo {
  route: RouteDecision;
  confidence: number;
  reason: string;
}
