import type { StudioSettings } from "./types";

export function requiresDiarization(settings: StudioSettings): boolean {
  return settings.diarization.length > 0 || settings.voiceCloning.length > 0;
}

