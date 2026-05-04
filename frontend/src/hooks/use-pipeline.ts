"use client";

import { useCallback, useReducer, useRef } from "react";
import type {
  PipelineStage,
  PipelineState,
  StageState,
  StudioSettings,
  Video,
  VideoVariant,
} from "@/lib/types";
import {
  diarizeVideo,
  downloadVideo,
  transcribeVideo,
  translateVideo,
  synthesizeSpeech,
  stitchVideo,
} from "@/lib/api";
import { computeConfigEntries, type ConfigEntry } from "@/lib/config-id";
import { requiresDiarization } from "@/lib/settings";

const STAGES: PipelineStage[] = [
  "download",
  "transcribe",
  "diarize",
  "translate",
  "tts",
  "stitch",
];

function initialStages(settings?: StudioSettings): Record<PipelineStage, StageState> {
  const diarizationEnabled = settings ? requiresDiarization(settings) : false;
  return Object.fromEntries(
    STAGES.map((s) => [
      s,
      s === "diarize" && !diarizationEnabled
        ? { status: "skipped" as const }
        : { status: "pending" as const },
    ])
  ) as Record<PipelineStage, StageState>;
}

const INITIAL_STATE: PipelineState = {
  status: "idle",
  stages: initialStages(),
  selectedStage: "download",
  variants: [],
};

function makeVariantId(videoId: string, configId: string): string {
  return `${videoId}::${configId}`;
}

type Action =
  | { type: "START"; videoId: string; settings: StudioSettings; configs: ConfigEntry[] }
  | { type: "STAGE_ACTIVE"; stage: PipelineStage }
  | { type: "STAGE_COMPLETE"; stage: PipelineStage; result: unknown; duration_ms: number; skipped?: boolean }
  | { type: "STAGE_ERROR"; stage: PipelineStage; error: string }
  | { type: "SELECT_STAGE"; stage: PipelineStage }
  | { type: "PIPELINE_COMPLETE" }
  | { type: "SELECT_VARIANT"; variantId: string }
  | { type: "RESET" };

function reducer(state: PipelineState, action: Action): PipelineState {
  switch (action.type) {
    case "RESET":
      return INITIAL_STATE;

    case "START": {
      const newVariants: VideoVariant[] = action.configs.map((cfg) => ({
        id: makeVariantId(action.videoId, cfg.id),
        sourceVideoId: action.videoId,
        configId: cfg.id,
        label: cfg.label,
        settings: action.settings,
        status: "processing" as const,
      }));
      const newVariantIds = new Set(newVariants.map((v) => v.id));
      return {
        ...state,
        status: "running",
        videoId: action.videoId,
        stages: initialStages(action.settings),
        selectedStage: "download",
        variants: [
          ...state.variants.filter((v) => !newVariantIds.has(v.id)),
          ...newVariants,
        ],
        activeVariantId: newVariants[0].id,
      };
    }

    case "STAGE_ACTIVE":
      return {
        ...state,
        stages: {
          ...state.stages,
          [action.stage]: { status: "active", started_at: Date.now() },
        },
        selectedStage: action.stage,
      };

    case "STAGE_COMPLETE":
      return {
        ...state,
        stages: {
          ...state.stages,
          [action.stage]: {
            status: action.skipped ? "skipped" : "complete",
            result: action.result,
            duration_ms: action.duration_ms,
          },
        },
        selectedStage: action.stage,
      };

    case "STAGE_ERROR":
      return {
        ...state,
        status: "error",
        stages: {
          ...state.stages,
          [action.stage]: { status: "error", error: action.error },
        },
        selectedStage: action.stage,
        variants: state.variants.map((v) =>
          v.id === state.activeVariantId ? { ...v, status: "error" as const } : v
        ),
      };

    case "PIPELINE_COMPLETE":
      return {
        ...state,
        status: "complete",
        selectedStage: "stitch",
        variants: state.variants.map((v) =>
          v.sourceVideoId === state.videoId && v.status === "processing"
            ? { ...v, status: "complete" as const }
            : v
        ),
      };

    case "SELECT_STAGE":
      return { ...state, selectedStage: action.stage };

    case "SELECT_VARIANT":
      return { ...state, activeVariantId: action.variantId };

    default:
      return state;
  }
}

export function usePipeline() {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);
  const runTokenRef = useRef(0);
  const runningRef = useRef(false);

  const invalidateActiveRun = useCallback(() => {
    runTokenRef.current += 1;
    runningRef.current = false;
  }, []);

  const selectStage = useCallback(
    (stage: PipelineStage) => dispatch({ type: "SELECT_STAGE", stage }),
    []
  );

  const selectVariant = useCallback(
    (variantId: string) => dispatch({ type: "SELECT_VARIANT", variantId }),
    []
  );

  const runPipeline = useCallback(async (video: Video, settings: StudioSettings) => {
    if (runningRef.current) {
      return;
    }

    runningRef.current = true;
    const runToken = runTokenRef.current + 1;
    runTokenRef.current = runToken;

    const dispatchIfCurrent = (action: Action) => {
      if (runTokenRef.current === runToken) {
        dispatch(action);
      }
    };

    const configs = computeConfigEntries(settings);
    dispatchIfCurrent({ type: "START", videoId: video.id, settings, configs });

    const run = async <T,>(
      stage: PipelineStage,
      fn: () => Promise<T>
    ): Promise<T> => {
      dispatchIfCurrent({ type: "STAGE_ACTIVE", stage });
      const t0 = performance.now();
      try {
        const result = await fn();
        const skipped = typeof result === "object" && result !== null && "skipped" in result
          ? (result as Record<string, unknown>).skipped === true
          : false;
        dispatchIfCurrent({
          type: "STAGE_COMPLETE",
          stage,
          result,
          duration_ms: Math.round(performance.now() - t0),
          skipped,
        });
        return result;
      } catch (err) {
        dispatchIfCurrent({
          type: "STAGE_ERROR",
          stage,
          error: err instanceof Error ? err.message : String(err),
        });
        throw err;
      }
    };

    try {
      const dl = await run("download", () => downloadVideo(video.url));
      await run("transcribe", () => transcribeVideo(dl.video_id, settings.useYoutubeCaptions));
      if (requiresDiarization(settings)) {
        await run("diarize", () => diarizeVideo(dl.video_id));
      }
      await run("translate", () => translateVideo(dl.video_id, "es"));

      // Run TTS + stitch for each config entry.
      // SELECT_VARIANT before each iteration so STAGE_ERROR marks the correct variant.
      for (const cfg of configs) {
        dispatchIfCurrent({ type: "SELECT_VARIANT", variantId: makeVariantId(video.id, cfg.id) });
        const alignment = cfg.dubbing === "aligned";
        await run("tts", () => synthesizeSpeech(dl.video_id, cfg.id, alignment));
        await run("stitch", () => stitchVideo(dl.video_id, cfg.id));
      }

      dispatchIfCurrent({ type: "PIPELINE_COMPLETE" });
    } catch {
      // Error already dispatched in run()
    } finally {
      if (runTokenRef.current === runToken) {
        runningRef.current = false;
      }
    }
  }, []);

  const reset = useCallback(() => {
    invalidateActiveRun();
    dispatch({ type: "RESET" });
  }, [invalidateActiveRun]);

  return { state, runPipeline, selectStage, selectVariant, reset };
}
