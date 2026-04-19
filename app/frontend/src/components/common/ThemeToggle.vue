<template>
  <button class="theme-toggle" @click="toggleTheme">
    <span class="theme-toggle__label">主题</span>
    <span class="theme-toggle__value">
      {{ currentThemeLabel }}
    </span>
  </button>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useTheme } from '@/composables/useTheme'

const { currentTheme, toggleTheme } = useTheme()

const currentThemeLabel = computed(() => {
  if (currentTheme.value === 'dark') return '深色'
  if (currentTheme.value === 'light') return '浅色'
  return '跟随系统'
})
</script>

<style scoped>
.theme-toggle {
  height: 40px;
  padding: 0 14px;
  border: 1px solid var(--border-color);
  border-radius: 12px;
  background: var(--surface-1);
  color: var(--text-primary);
  display: inline-flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  transition:
    background 160ms ease,
    border-color 160ms ease,
    color 160ms ease;
}

.theme-toggle:hover {
  background: var(--surface-2);
  border-color: var(--border-strong);
}

.theme-toggle__label {
  color: var(--text-secondary);
  font-size: 13px;
}

.theme-toggle__value {
  font-size: 13px;
  font-weight: 600;
}
</style>
