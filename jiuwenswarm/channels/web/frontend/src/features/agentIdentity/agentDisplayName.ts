// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import { useSyncExternalStore } from 'react';

// 单智能体回复上方的名字：config.get 的 agent_display_name（identity.name），
// 未设置时为 Jiuwen。
export const DEFAULT_AGENT_DISPLAY_NAME = 'Jiuwen';

let displayName = DEFAULT_AGENT_DISPLAY_NAME;
const listeners = new Set<() => void>();

export function setAgentDisplayName(value: unknown): void {
  const next = (typeof value === 'string' ? value.trim() : '') || DEFAULT_AGENT_DISPLAY_NAME;
  if (next === displayName) return;
  displayName = next;
  listeners.forEach((listener) => listener());
}

export function getAgentDisplayName(): string {
  return displayName;
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useAgentDisplayName(): string {
  return useSyncExternalStore(subscribe, getAgentDisplayName, getAgentDisplayName);
}
